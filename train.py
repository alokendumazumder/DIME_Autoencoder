import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import argparse
from tqdm import tqdm
import model
import utils
import pandas as pd
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.manifold import TSNE
# from mine.models.mine import Mine,T
from copy import deepcopy

parser = argparse.ArgumentParser(description="Training Autoencoders")

parser.add_argument('--n', type=int, help='latent dimension', default=64)

parser.add_argument('--t', type=float, help='uniforms t', default=0.005) #NO WORK

parser.add_argument('--epochs', type=int, help='#epochs', default=100)

parser.add_argument('--dataset', type=str, default="celeba")

parser.add_argument('--optimizer', type=str, default="adam")

parser.add_argument('--unif_lambda', type=float, help='Nuc lambda index [0.001, 0.01, 0.1, 0.5, 1, 5]', default=1e-2)

parser.add_argument('--vanilla', action='store_true', help='VANILLA')

# Let following arguments be in default
parser.add_argument('--batch_size', type=int, default=100) #32 earlier
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--checkpoint', type=str, default="./checkpoint/")
parser.add_argument('--data-path', type=str, default="./data/")
parser.add_argument('--df-name', type=str, default="celeba_dime.csv")

class L2Norm(torch.nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)

def uniform(x,t):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def compute_gradient_norm(model):
    """
    Compute the L2 norm of gradients for all parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        float: L2 norm of all gradients
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def plot_gradient_variance(epoch_losses, save_path):
    """
    Plot the variance of total gradient norms across batches for each epoch using matplotlib.
    
    Args:
        epoch_losses: List containing epoch information with batch-wise gradient data
        save_path: Path to save the plot
    """
    epochs = [epoch_data['epoch'] for epoch_data in epoch_losses]
    grad_variances = [epoch_data['total_grad_variance'] for epoch_data in epoch_losses]
    
    if not epochs:
        print("No gradient variance data to plot")
        return
    
    # Create the plot using matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, grad_variances, 'b-o', linewidth=2, markersize=6, alpha=0.8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Gradient Variance', fontsize=12)
    plt.title('Variance of Total Gradient Norms Across Batches', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_path, 'gradient_variance_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Gradient variance plot saved to: {plot_path}")

def plot_latent_space(latent_vectors, labels, epoch,args):
    """
    Plots an interactive 3D scatter plot of the latent space with different colors for each class.
    Args:
        latent_vectors (Tensor): The latent representations of the inputs.
        labels (Tensor): The labels for each point.
        epoch (int): The current epoch number.
    """
    norm = L2Norm()
    # Ensure the latent vectors and labels are on the CPU and converted to numpy arrays
    latent_vectors = latent_vectors.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    # Reduce dimension to 3 if latent_vectors have more than 3 dimensions
    if latent_vectors.shape[1] > 3:
        tsne = TSNE(n_components=3, random_state=42)
        latent_vectors = tsne.fit_transform(latent_vectors)
        latent_vectors = torch.tensor(latent_vectors)
        latent_vectors = norm(latent_vectors)
        latent_vectors = latent_vectors.cpu().detach().numpy()

    # Create a DataFrame for easier plotting
    df = pd.DataFrame(latent_vectors, columns=['Latent Dimension 1', 'Latent Dimension 2', 'Latent Dimension 3'])
    df['Label'] = labels

    # Create the plotly scatter plot
    fig = px.scatter_3d(df, x='Latent Dimension 1', y='Latent Dimension 2', z='Latent Dimension 3',
                        color='Label', title=f'3D Latent Space at Epoch {epoch}',
                        color_continuous_scale=px.colors.qualitative.Alphabet,
                        labels={'Latent Dimension 1': 'Latent Dimension 1',
                                'Latent Dimension 2': 'Latent Dimension 2',
                                'Latent Dimension 3': 'Latent Dimension 3'})

    # Save the plot as an HTML file
    fig.write_html(f'plots/latent_space_3d_epoch_mnist_latent_dim_{args.n}_uniform_lambda_{args.unif_lambda}_{epoch}.html')

def main(args):

    # use gpu ##########################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataset ##########################################
    if args.dataset == "mnist":
        args.data_path = args.data_path + "mnist/"
        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)

        train_set = datasets.MNIST(args.data_path, train=True,
                                   download=True,
                                   transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))
        valid_set = datasets.MNIST(args.data_path, train=False,
                                   download=True,
                                   transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))
        
    elif args.dataset == "fmnist":
        args.data_path = args.data_path + "fmnist/"
        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)

        train_set = datasets.FashionMNIST(args.data_path, train=True,
                                   download=True,
                                   transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))
        valid_set = datasets.FashionMNIST(args.data_path, train=False,
                                   download=True,
                                   transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))
    elif args.dataset == "intel":
        train_set = ImageFolder(
            args.data_path + 'intel/train/',
            transform=transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
        valid_set = ImageFolder(
            args.data_path + 'intel/test/',
            transform=transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
    
    elif args.dataset == "cifar10":
        train_set = datasets.CIFAR10(
            args.data_path + '/cifar10/',
            train=True,
            download=True,
            transform=transforms.Compose([transforms.Resize([32, 32]),
                                          transforms.ToTensor()]))
        valid_set = datasets.CIFAR10(
            args.data_path + '/cifar10/',
            train=False,
            download=True,
            transform=transforms.Compose([transforms.Resize([32, 32]),
                                          transforms.ToTensor()]))

    elif args.dataset == "celeba":
        train_set = utils.ImageFolder(
            args.data_path + 'celeba/train/',
            transform=transforms.Compose([transforms.CenterCrop(148), #erlier 148
                                          transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
        valid_set = utils.ImageFolder(
            args.data_path + 'celeba/val/',
            transform=transforms.Compose([transforms.CenterCrop(148),
                                          transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
        
    elif args.dataset == "shape":
        train_set = utils.ShapeDataset(
            data_size=50000)
        valid_set = utils.ShapeDataset(
            data_size=10000)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=32,
        batch_size=args.batch_size
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        num_workers=32,
        batch_size=args.batch_size
    )

    # init networks ##########################################
    net = model.AE(args)
    net.to(device)
    
    # n_lambda= [0.001, 0.005, 0.003, 0.0001, 0.0005, 0.00001]
    
    # Define the nuclear norm regularization strength
    uniform_strength = args.unif_lambda
    
    # optimizer ##########################################
    if args.optimizer=='adam':
        optimizer = optim.Adam(net.parameters(), args.lr)
    elif args.optimizer=='sgd':
        optimizer= optim.SGD(net.parameters(), args.lr)
    elif args.optimizer=='adagrad':
        optimizer= optim.Adagrad(net.parameters(), args.lr)

    # mine = Mine(T=T(x_dim = (1*32*32),z_dim = (args.n)).to(device)).to(device) #mnist
    
    # Initialize lists to store loss and gradient information
    epoch_losses = []
    
    # train ################################################
    save_path = args.checkpoint + "/" + args.dataset + "/"
    logs_path = save_path + "logs/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    for e in range(args.epochs):
        
        recon_loss = 0
        loss_mse = 0
        loss_uni = 0
        
        # Lists to store batch-wise information for this epoch
        batch_losses = []
        batch_total_grad_norms = []  # Store total gradient norms for variance calculation
        
        z_all = []
        x_all = []
        for yi, l, in tqdm(train_loader):
            net.train()

            optimizer.zero_grad()

            yi = yi.to(device)
            mse_loss, z = net(yi)

            # mat= net.mlp.hidden[0].weight
            # print(z.norm(p=2,dim=1,keepdim=True))
            # Compute the nuclear norm regularization term
            # nuclear_norm_regularization = torch.linalg.norm(mat, ord='nuc')
            #unif reg
            
            loss_mse += mse_loss.item()
            uniform_loss = uniform(z, args.t)
            loss_uni += uniform_loss.item()

            params = [p for p in net.parameters() if p.requires_grad]
            mse_grad = torch.autograd.grad(mse_loss, params, retain_graph=True, create_graph=False, allow_unused=True)
            mse_grad_norm  = torch.sqrt(sum((g.detach()**2).sum() for g in mse_grad if g is not None))
            optimizer.zero_grad() 
            unifiorm_norm = torch.autograd.grad(uniform_strength * uniform_loss, params, retain_graph=True, create_graph=False, allow_unused=True)
            uniform_grad_norm = torch.sqrt(sum((g.detach()**2).sum() for g in unifiorm_norm if g is not None))
            optimizer.zero_grad() 

            # Calculate total loss
            total_loss = mse_loss + uniform_strength * uniform_loss
            recon_loss += total_loss.item()
            
            # Backward pass
            total_loss.backward()
            
            # Compute gradient norms before optimizer step
            total_grad_norm = compute_gradient_norm(net)
            
            # Store batch information
            batch_info = {
                'epoch': e,
                'mse_loss': mse_loss.item(),
                'uniform_loss': uniform_loss.item(),
                'weighted_uniform_loss': (uniform_strength * uniform_loss).item(),
                'total_loss': total_loss.item(),
                'mse_grad_norm': mse_grad_norm.item(),
                'uniform_grad_norm': uniform_grad_norm.item(),
                'total_grad_norm': total_grad_norm,
                'uniform_strength': uniform_strength
            }
            batch_losses.append(batch_info)
            batch_total_grad_norms.append(total_grad_norm)  # Store for variance calculation
            
            optimizer.step()
            
            # if ((e+1)%10==0):
            #     z_all.append(z.detach().to(device))
            #     x_all.append(yi.to(device))

        # Calculate epoch averages
        loss_uni /= len(train_loader)
        loss_mse /= len(train_loader)
        recon_loss /= len(train_loader)
        
        # Calculate average gradient norms for the epoch
        # avg_mse_grad_norm = sum([b['mse_grad_norm'] for b in batch_losses]) / len(batch_losses)
        # avg_uniform_grad_norm = sum([b['uniform_grad_norm'] for b in batch_losses]) / len(batch_losses)
        # avg_total_grad_norm = sum([b['total_grad_norm'] for b in batch_losses]) / len(batch_losses)
        
        # Calculate gradient variance for this epoch
        # total_grad_variance = torch.var(torch.tensor(batch_total_grad_norms)).item() if len(batch_total_grad_norms) > 1 else 0.0
        
        # Store epoch information
        # epoch_info = {
        #     'epoch': e,
        #     'avg_mse_loss': loss_mse,
        #     'avg_uniform_loss': loss_uni,
        #     'avg_weighted_uniform_loss': uniform_strength * loss_uni,
        #     'avg_total_loss': recon_loss,
        #     'avg_mse_grad_norm': avg_mse_grad_norm,
        #     'avg_uniform_grad_norm': avg_uniform_grad_norm,
        #     'avg_total_grad_norm': avg_total_grad_norm,
        #     'uniform_strength': uniform_strength
        # }
        # epoch_losses.append(epoch_info)
        
        # print(f"Epoch {e}")
        # print(f"  MSE Loss: {loss_mse:.6f} (grad norm: {avg_mse_grad_norm:.6f})")
        # print(f"  Uniform Loss: {loss_uni:.6f} (grad norm: {avg_uniform_grad_norm:.6f})")
        # print(f"  Weighted Uniform Loss: {uniform_strength * loss_uni:.6f}")
        # print(f"  Total Loss: {recon_loss:.6f} (grad norm: {avg_total_grad_norm:.6f})")
        # print(f"  Total Gradient Variance: {total_grad_variance:.6f}")

        # save model #########################################
        if ((e+1)%50==0):
            torch.save(net.state_dict(), save_path+ 't' + str(args.t) + '_n' + str(args.n) + '_' + str(args.optimizer) + '_uniform_lambda' + str(args.unif_lambda) + "_" + str(e+1) + "_lr" + str(args.lr)+"_no_normalized")
    
    # Save epoch-wise losses
    # epoch_df = pd.DataFrame(epoch_losses)
    # epoch_df.to_csv(f"{logs_path}_epoch_losses.csv", index=False)
    
    # # Plot gradient variance
    # # plot_gradient_variance(epoch_losses, logs_path)
    
    # # Update the main results file
    # df = pd.read_csv(args.df_name)
    # new_row = pd.DataFrame([{
    #     "dataset": args.dataset,
    #     "LR": args.lr, 
    #     "latent_dim": args.n, 
    #     "T": args.t,
    #     "lambda": args.unif_lambda,
    #     "unif_loss": loss_uni,
    #     "lambda*unif_loss": uniform_strength * loss_uni,
    #     "total_loss": recon_loss,
    #     "mse_grad_norm": avg_mse_grad_norm,
    #     "uniform_grad_norm": avg_uniform_grad_norm,
    #     "total_grad_norm": avg_total_grad_norm,
    #     "total_grad_variance": total_grad_variance
    # }])
    # df = pd.concat([df, new_row], ignore_index=True)
    # df.to_csv(args.df_name, index=False)
    
    # print(f"\nTraining completed!")
    # print(f"Logs saved to: {logs_path}")
    # print(f"- epoch_losses.csv: Contains epoch-wise averages")
    # print(f"- batch_losses_epoch_X.csv: Contains batch-wise details (saved every 10 epochs)")
    # print(f"- gradient_variance_plot.png: Gradient variance plot across epochs")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)