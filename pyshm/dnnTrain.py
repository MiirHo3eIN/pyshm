import matplotlib.pyplot as plt
import torch 
from torch import nn
import numpy as np 
import seaborn as sns
import os
import time 


from .utils import generate_hexadecimal, write_to_csv


# DO NOT change the following configurations. These features are not implemented yet.
PLOT  = False 
device = "cuda" if torch.cuda.is_available() else "cpu" 

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        err = abs((validation_loss - self.min_validation_loss))
        self.min_validation_loss = validation_loss

        if err >= self.min_delta:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def plot_double(ax, x_org, x_rec, sensor, label):
    
    ax.plot(x_org[0, sensor, :].cpu().detach().numpy(), label = 'original data', color = "green")
    ax.plot(x_rec[0, sensor, :].cpu().detach().numpy(), label = label, color = "black")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Cp data")
    ax.set_title(f"Sensor {sensor} | Reconstructed vs original data")

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel("Cp Error", color=color)  # 50% transparent
    ax2.plot(np.subtract(x_org[0, sensor, :].cpu().detach().numpy(),
                         x_rec[0, sensor, :].cpu().detach().numpy()), 
                         color=color, linestyle="dashed", alpha=0.4)

    ax.legend()

def plot_original_vs_reconstructed(x_org: torch.tensor, x_rec: torch.tensor, sensor: list, label:str, path:str ) -> None:
    with sns.plotting_context("poster"):
        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        plot_double(ax[0], x_org, x_rec, sensor[0], label)    
        plot_double(ax[1], x_org, x_rec, sensor[1], label)
        plt.tight_layout()

        filename = "ephoc_" + str(label) + "_original_vs_reconstructed.png"
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')

        plt.show(block=False)
        plt.pause(3)
        plt.close()


class Train(nn.Module):

    def __init__(self, model: object, 
                 epochs: int, 
                 alpha:float, 
                 output_filter:bool, 
                 path:str, 
                 early_stop: bool = True,
                 pre_trained_path: str = "",
                 reconstruction_loss: str = "mse",
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.model = model.to(device)
        self.epochs = epochs
        self.start_epoch = 0 # if we want to start from a specific epoch
        self.alpha = alpha
        self.output_filter = output_filter
        self.path = path
        self.e_stop = early_stop
        self.best_valid_loss = float('inf')
        self.best_checkpoint_path = os.path.join(path, "best_checkpoint.pt")
        self.reconstruction_loss = nn.MSELoss() #if reconstruction_loss == "mse"
        if self.e_stop == True:
            self.early_stopper = EarlyStopper(patience=3, min_delta=5)


        print("-"*50)
        print("Setup Training...")
        lr = 0.0001
        betas = (0.9, 0.98)
        eps = 1e-8

        # Leave these two as we may use them for the future models
        # criterion_method = ("mse", "smooth_l1")    
        # reduction_type = "sum"
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
        self.criterion = self.reconstruction_loss

        if pre_trained_path != "":
            #pre-trained model
            self.load_checkpoint(pre_trained_path)
            self.model.train()


        print(f"Used Device: {device}")
        print(f"Optimizer | lr: {lr} | betas: {betas} | eps: {eps}")
        print(f"Criterion alpha: {self.alpha}")
        print("\n")

    def save_checkpoint(self, epoch, loss):
    
        model_number = generate_hexadecimal()
   
        #save checkpoint
        # Additional information
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
        }, self.best_checkpoint_path)  
                # }, f"{self.path}{epoch}_checkpoint.pt")


        return model_number 
    
    def load_checkpoint(self, path: str):

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        self.best_valid_loss = checkpoint['loss']

        print("\n")
        print("Model loaded from pre-trained, Epoch: " + str(self.start_epoch) + " Loss: " + str(loss))
        print("\n")

        return  loss
   
    def forward_propagation(self, x_batch, y_batch, idx, epoch ):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_train = self.model.forward(x_batch.float())

        train_loss = self.criterion(y_train.float(), y_batch.float())

        # @ahmad-mirsalari: You can uncomment this to plot but needs to be modified. 
        # # Give me a few days to fix this.         
        if idx == 1 and epoch%10 == 0 and PLOT == 1:       
            plot_original_vs_reconstructed(y_batch, y_train, sensor = [5, 15], label = f"Epoch {epoch} | train", path = self.path)
        
        return train_loss

    def back_propagation(self, train_loss):
    
        # Backpropagation
        self.optimizer.zero_grad()
        train_loss.backward()
        # gradient clipping is helpful sometimes to avoid exploding gradients. 
        # @ahmad-mirsalari: You can uncomment this to clip the gradients based on your observation and results.
        # max_norm = 0.0005
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        # # Update the parameters
        self.optimizer.step()


    def evaluate(self, x_batch, y_batch, idx, epoch):
        
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.no_grad():
            self.model.eval()
            y_valid = self.model.forward(x_batch.float())

            # @ahmad-mirsalari: You can uncomment this to plot but needs to be modified. 
            # Give me a few days to fix this. 
            if idx == 1 and epoch%10 == 0 and PLOT == 1:
                plot_original_vs_reconstructed(y_batch, y_valid, sensor = [5, 15], label = f"Epoch {epoch} | validation", path = self.path)
            valid_loss = self.criterion(y_valid.float(), y_batch.float())

            return valid_loss


    def forward(self, train_x, valid_x, test_x):
        
        
        train_epoch_loss, valid_epoch_loss = [], []
        train_batch_loss, valid_batch_loss = [], []
        train_total_loss, valid_total_loss = [], []
        
        print("-"*50)
        print("Starting Training...")
        time_start = time.time()
        print(time_start)

        
         
        for epoch in np.arange(self.start_epoch, self.epochs):
            #print("Inside the for loop")
            print(f"Epoch: {epoch+1}/{self.epochs}")

            ### TRAINING PHASE ###
            idx = 0
            for x_batch, y_batch in (train_x):
                # print("x_batch shape: ", x_batch.shape)
                # print("y_batch shape: ", y_batch.float().shape)
                idx += 1
                train_loss = self.forward_propagation(x_batch, y_batch, idx = idx, epoch = epoch)

                train_batch_loss += [train_loss]
                train_epoch_loss += [train_loss.item()]
                self.back_propagation(train_loss)

            idx = 0 
            ### VALIDATION PHASE ###
            for x_batch, y_batch in (valid_x):
                    idx += 1 
                    valid_loss = self.evaluate(x_batch, y_batch, idx = idx, epoch = epoch)
                    
                    valid_batch_loss += [valid_loss]
                    valid_epoch_loss += [(valid_loss.item())]

            
            # print(f"\t Train loss = {sum(train_epoch_loss)/len(train_epoch_loss):.05}, \
            #         Validation Loss = {sum(valid_epoch_loss)/len(valid_epoch_loss):.05}")

            # train_total_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))
            # valid_total_loss.append(sum(valid_epoch_loss)/len(valid_epoch_loss))

            #save checkpoint
            # self.save_checkpoint(epoch, sum(train_epoch_loss)/len(train_epoch_loss))
            # Save checkpoint if validation loss has decreased
            
            avg_train_loss = sum(train_epoch_loss) / len(train_epoch_loss)
            avg_valid_loss = sum(valid_epoch_loss) / len(valid_epoch_loss)

            print(f"\t Train loss = {avg_train_loss:.05}, Validation Loss = {avg_valid_loss:.05}")

            train_total_loss.append(avg_train_loss)
            valid_total_loss.append(avg_valid_loss)

            # Save checkpoint if validation loss has decreased
            if avg_valid_loss < self.best_valid_loss:
                self.best_valid_loss = avg_valid_loss
                self.save_checkpoint(epoch+1, avg_valid_loss)



            #early stop condition
            if self.e_stop == True and self.early_stopper.early_stop(sum(train_epoch_loss)/len(train_epoch_loss)):
                print("Early stopping triggered.")
                break

            train_epoch_loss, valid_epoch_loss = [], []
            train_batch_loss, valid_batch_loss = [], []
            idx = 0

        
        time_end = time.time()
        train_time = time_end - time_start
        
        return train_time, train_total_loss, valid_total_loss
    



def save_model(model, path, arch_id, train_loss, valid_loss):
    
    model_number = generate_hexadecimal()
   
    torch.save(model.state_dict(), f"{path}{model_number}.pt")

    np.save(f"{path}{model_number}_train_loss.npy", np.array(train_loss))
    np.save(f"{path}{model_number}_valid_loss.npy", np.array(valid_loss))
    print(f"Saved the model: {model_number} with architecture Id: {arch_id}")

    return model_number 

def save_training_results(train_config):
    
    data = train_config.__dict__
    write_to_csv(data)
    print("Saved the training results to the csv file.")




def plot_loss(train_loss, valid_loss, model_id, path:str):
    
    with sns.plotting_context('poster'): 
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(train_loss, color = 'green', label = 'Train ')
        ax.plot(valid_loss, color = 'red', label = 'Valid ')

        ax.set_xlabel('epochs')
        ax.set_ylabel('Loss')

        ax.set_title(f"Loss trend for Model {model_id}")
        ax.legend()
        plt.tight_layout()

        filename = "plot_loss.png"
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')

        plt.show(block=False)
        plt.pause(3)
        plt.close()