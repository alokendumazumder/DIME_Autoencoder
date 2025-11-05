import os
import numpy as np
import argparse
import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn import mixture
import matplotlib.pyplot as plt
import models_test as models_test
import model_vae
from PIL import Image
import utils as utils
from scipy.stats import gaussian_kde
from pythae.samplers import VAMPSampler,HypersphereUniformSampler
import pickle
from sklearn.manifold import TSNE


import torch.nn as nn
from pythae.models import AutoModel
# from GMM_DAE.model import VAE as gmm_vae
# from irmae.model import AE as irame_ae
# from LoRAE_WACV24.model import AE as lorae_ae

parser = argparse.ArgumentParser(description="Generative and Downstream Tasks")
parser.add_argument('--dataset', type=str, default="celeba")
parser.add_argument('--name', type=str, default="CELEB")
parser.add_argument('--task', type=str, default="mvg")
parser.add_argument('--fid', action='store_true', help='Calculate FID score')
parser.add_argument('--org', action='store_true', help='Calculate FID score')
parser.add_argument('--n', type=int, help='latent dimension', default=128)
parser.add_argument('--t', type=float, help='layers', default=0.005)
parser.add_argument('--l', type=int, help='layers', default=4)
parser.add_argument('--vae', action='store_true', help='VAE')
 
parser.add_argument('--vanilla', action='store_true', help='VANILLA')

parser.add_argument('--irmae', action='store_true', help='IRMAE')
parser.add_argument('--gmm', action='store_true', help='Gmm')
parser.add_argument('--lorae', action='store_true', help='Lorae')

parser.add_argument("--bench_mark",action="store_true",help="Benchmarks")


parser.add_argument('--optimizer', type=str, default="adam")
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--unif_lambda', type=float, help='Uniform Lambda', default=1e-3)

parser.add_argument('--d', type=int, default=10)
parser.add_argument('-X', type=int, default=10)
parser.add_argument('-Y', type=int, default=10)
parser.add_argument('-N', type=int, default=100)
parser.add_argument('--test-size', type=int, default=100)
parser.add_argument('--batch',type=int,default=10,help='Which batch to take for generation')

parser.add_argument('--sample', type=int, default=10000)
parser.add_argument('--do_sample', action='store_true', help='DOES SAMPLING')
parser.add_argument('--model', type=str, default="huae")
parser.add_argument('--model_path', type=str, default="huae")
parser.add_argument('--checkpoint', type=str, default="/mnt/SSD_2/Alok/encoder/DIME/models")
parser.add_argument('--data-path', type=str, default="../data/")
parser.add_argument('--save-path', type=str, default="./results/")

class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)

def main(args):
    # np.seed(42)
    # dataset ##########################################
    if args.dataset == "mnist2k":
        args.data_path = args.data_path + "mnist/"
        if args.bench_mark:
            with open(f"{args.data_path}mnist2k_test_pythae.pkl","rb") as f:
                test_set = pickle.load(f)
        else:
            with open(f"{args.data_path}mnist2k_test_our.pkl","rb") as f:
                test_set = pickle.load(f)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=args.test_size
        )
    if args.dataset == "mnist":
        args.data_path = args.data_path + "mnist/"
        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)

        test_set = datasets.MNIST(args.data_path, train=False,download=True,
                                  transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))

        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=args.test_size
        )
    elif args.dataset == "fmnist":
        args.data_path = args.data_path + "fmnist/"
        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)

        test_set = datasets.FashionMNIST(args.data_path, train=False,
                                  transform=transforms.Compose([
                                     transforms.Resize([32, 32]),
                                     transforms.ToTensor()]))

        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=args.test_size
        )
    elif args.dataset == "intel":
        test_set = datasets.ImageFolder(
            args.data_path + 'intel/test/',
            transform=transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
        
        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=args.test_size
        )
    elif args.dataset == "cifar10":
        test_set = datasets.CIFAR10(
            args.data_path + 'cifar10/',
            train=False,
            transform=transforms.Compose([transforms.Resize([32, 32]),
                                          transforms.ToTensor()]))
        
        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=args.test_size
        )

    elif args.dataset == "celeba":
        test_set = utils.ImageFolder(
            args.data_path + 'celeba/test/',
            transform=transforms.Compose([transforms.CenterCrop(148),
                                          transforms.Resize([64, 64]),
                                          transforms.ToTensor()]))
        # test_set = test_set[:10000]
    elif args.dataset == "shape":
        test_set = utils.ShapeDataset(
            data_size=10000)
        
    if args.do_sample:
        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=args.test_size,
            sampler=torch.utils.data.SubsetRandomSampler(
            range(args.sample))
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            test_set,
            num_workers=32,
            batch_size=args.test_size
            )
    if args.dataset=="mnist2k":
        args.dataset = "mnist"
    # load model ##########################################
    if args.vae:
        net= model_vae.AE(args)
    elif args.bench_mark:
        net = AutoModel.load_from_folder(os.path.join(args.model_path, 'final_model'))
    elif args.irmae:
        net = irame_ae(args)
    elif args.gmm:
        try:
            from configparser import ConfigParser
        except ImportError:
            from configparser import ConfigParser  # ver. < 3.0

        major_idx = args.name
        config = ConfigParser()
        config.read('./GMM_DAE/config.ini')

        dataset = config.get(major_idx, 'dataset')
        img_size = config.getint(major_idx, 'image_size')
        epochs = config.getint(major_idx, 'epochs')
        latent_dim = config.getint(major_idx, 'latent_dim')
        image_num_channels = config.getint(major_idx, 'image_num_channels')
        nef = config.getint(major_idx, 'nef')
        ndf = config.getint(major_idx, 'ndf')
        latent_noise_scale = config.getfloat(major_idx, 'latent_noise_scale')

        # Initialize the model.
        net = gmm_vae(dataset=dataset, nc=image_num_channels, ndf=ndf, nef=nef,
                    nz=latent_dim, isize=img_size, latent_noise_scale=latent_noise_scale)
    elif args.lorae:
        net = lorae_ae(args)
    else:
        net = models_test.AE(args)
    
    path = args.save_path + args.dataset + "/" + args.task
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    if args.vae:
        net.load_state_dict(torch.load(args.checkpoint + "/" + args.dataset
                        + '/vae',
                        map_location=torch.device('cpu')))
        path += '/vae' 

    elif args.vanilla:
        net.load_state_dict(torch.load(args.checkpoint + "/" + args.dataset
                        + '/vanilla_'+str(args.n),
                        map_location=torch.device('cpu')))
        path += '/vanilla' 
    
    elif args.irmae or args.lorae or args.gmm:
        state_dict = torch.load(args.checkpoint, map_location="cpu",weights_only=True)

        # Detect if keys are prefixed with "module."
        if any(k.startswith("module.") for k in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
            state_dict = new_state_dict
        print(state_dict.keys())
        net.load_state_dict(state_dict)

    elif args.bench_mark:
        pass
    else:
        net.load_state_dict(torch.load(args.checkpoint + "/" + args.dataset + "/"
                        + 't' + str(args.t) + '_n' + str(args.n) + '_' + str(args.optimizer) + '_unfiorm_lambda' + str(args.unif_lambda),
                        # + "_100_lr" + str(args.lr),
                        map_location=torch.device('cpu'))) 
        # path += '/' + 't' + str(args.t) + '_n' + str(args.n) + '_' + str(args.optimizer) + '_nuc_lambda_index' + str(args.unif_lambda)
    
    
    net.eval()
    norm = L2Norm()
    fig, axs = plt.subplots(args.X, args.Y, figsize=[args.Y, args.X])
    i = 0
    if args.task == "reconstruction":
        y_hat = []
        y_org = []
        i = 0
        test_iter = iter(test_loader)  # Create the iterator only once
        yi, _ = None, None  # Initialize variables

        for i in range(args.batch):  # Advance the iterator args.batch times
            yi, _ = next(test_iter)
        # print("shape of yi is" + yi.shape)
        if args.bench_mark:
            zi = net.encoder(yi).embedding
        else:
            zi = net.encode(yi)
        y_hat = net.decode(zi).data.numpy()
        # for yi,_ in test_loader:
        #     y_org = list(yi.numpy())
        #     if args.bench_mark:
        #         zi = net.encoder(yi).embedding
        #     else:
        #         zi = net.encode(yi)
        #         # zi = norm(zi)
        #     if args.bench_mark:
        #         y_hat_i= net.decoder(zi).reconstruction.data.numpy()
        #     else:
        #         y_hat_i= net.decode(zi).data.numpy()
        #     y_hat.append(y_hat_i)
        #     i += 1
        #     if args.fid:
        #         if i==100: break
        #     elif i>args.batch:
        #         break
        # y_hat = np.concatenate(y_hat, axis=0)
    elif args.task == "interpolation":
        # print(len(test_loader))
        test_iter = iter(test_loader)  # Create the iterator only once
        yi, _ = None, None  # Initialize variables

        for i in range(args.batch):  # Advance the iterator args.batch times
            yi, _ = next(test_iter)
        # print("shape of yi is" + yi.shape)
        if args.bench_mark:
            zi = net.encoder(yi).embedding
        else:
            zi = net.encode(yi)
        
        zs = []
        for i in range(args.X):
            z0 = zi[i*2]
            z1 = zi[i*2+1]

            for j in range(args.Y):
                zs.append((z0 - z1) * j / args.Y + z1)
        zs = torch.stack(zs, axis=0)
        if not args.bench_mark:
            zs = norm(zs)
        if args.bench_mark:
            y_hat= net.decoder(zs).reconstruction.data.numpy()
        else:
            y_hat= net.decode(zs).data.numpy()

    elif args.task == "mvg":
        z = []
        for yi, _ in test_loader:
            # print(yi)
            if args.bench_mark:
                zi = net.encoder(yi).embedding
            else:
                zi = net.encode(yi)
            z.append(zi.detach().numpy())
        z = np.concatenate(z, axis=0)
        mu = np.average(z, axis=0)
        sigma = np.cov(z, rowvar=False)

        # generate corresponding sample z
        if args.fid:
            if args.dataset=='intel':
                zs = np.random.multivariate_normal(mu, sigma, 3000)
            elif args.dataset=='cifar10':
                zs = np.random.multivariate_normal(mu, sigma, 10000)
            elif args.dataset=='shape':
                zs = np.random.multivariate_normal(mu, sigma, 10000)
            elif args.dataset=='mnist' or args.dataset=='fmnist':
                zs = np.random.multivariate_normal(mu, sigma, 10000)
            elif args.dataset=='celeba':
                zs = np.random.multivariate_normal(mu, sigma, args.sample)
        else:
            zs = np.random.multivariate_normal(mu, sigma, args.X * args.Y)
        
        zs = torch.Tensor(zs)
        # if not args.bench_mark:
            # zs = norm(zs)

        if args.fid==True:
            y_hat= []
            print(zs.shape[0],args.test_size)
            for i in range(int(zs.shape[0]/args.test_size)):
                
                k= i*args.test_size
                if args.bench_mark:
                    y_temp= net.decoder(zs[k:k+args.test_size , :]).reconstruction.data.numpy()
                else:
                    y_temp= net.decode(zs[k:k+args.test_size , :]).data.numpy()

                y_hat.append(y_temp)
            
            y_hat = np.concatenate(y_hat, axis=0)
            
        else:
            if args.bench_mark:
                y_hat= net.decoder(zs).reconstruction.data.numpy()
            else:
                y_hat= net.decode(zs).data.numpy()
    

    elif args.task == "stdn":
        z = []
        for yi, _ in test_loader:
            if args.bench_mark:
                zi = net.encoder(yi)['embedding']
            else:
                zi = net.encode(yi)
            # _,zi = net.forward(yi)
            z.append(zi.detach().numpy())
        z = np.concatenate(z, axis=0)
        mu = np.average(z, axis=0)
        sigma = np.cov(z, rowvar=False)
        mu_0 = np.zeros((len(mu)), dtype=float)
        sigma_0 = np.identity(len(sigma),dtype=float)
        # generate corresponding sample z
        if args.fid:
            if args.dataset=='intel':
                zs = np.random.multivariate_normal(mu_0, sigma_0, 3000)
            elif args.dataset=='cifar10':
                zs = np.random.multivariate_normal(mu_0, sigma_0, 10000)
            elif args.dataset=='shape':
                zs = np.random.multivariate_normal(mu_0, sigma_0, 10000)
            elif args.dataset=='mnist' or args.dataset=='fmnist':
                zs = np.random.multivariate_normal(mu_0, sigma_0, 10000)
            elif args.dataset=='celeba':
                zs = np.random.multivariate_normal(mu_0, sigma_0, args.sample)
        else:
            zs = np.random.multivariate_normal(mu_0, sigma_0, args.X * args.Y)
        
        zs = torch.Tensor(zs)
        # if not args.bench_mark:
        #     zs = norm(zs)

        if args.fid==True:

            y_hat= []

            for i in range(int(zs.shape[0]/args.test_size)):
                
                k= i*args.test_size

                if args.bench_mark:
                    y_temp= net.decoder(zs[k:k+args.test_size , :]).reconstruction.data.numpy()
                else:
                    y_temp= net.decode(zs[k:k+args.test_size , :]).data.numpy()

                y_hat.append(y_temp)
            
            y_hat = np.concatenate(y_hat, axis=0)
            
        else:
            if args.bench_mark:
                y_hat= net.decoder(zs).reconstruction.data.numpy()
            else:
                y_hat= net.decode(zs).data.numpy()

    elif args.task == "gmm":
        z = []
        for yi,_ in (test_loader):
            if args.bench_mark:
                zi = net.encoder(yi)['embedding']
            else:
                zi = net.encode(yi)
            z.append(zi.detach().numpy())
        z = np.concatenate(z, axis=0)
        gmm = mixture.GaussianMixture(
            n_components=args.d, covariance_type='full')

        gmm.fit(z)
        
        if args.fid:

            if args.dataset=='intel':
                zs, _ = gmm.sample(3000)
            elif args.dataset=='cifar10':
                zs,_ = gmm.sample(10000)
            
            elif args.dataset=='shape':
                zs,_ = gmm.sample(10000)
            
            elif args.dataset=='mnist' or args.dataset=='fmnist':
                zs,_ = gmm.sample(10000)
            elif args.dataset=='celeba':
                zs,_ = gmm.sample(args.sample)

        else:
            zs, _ = gmm.sample(args.X * args.Y)

        zs = torch.Tensor(zs)
        # if not args.bench_mark:
        #     zs = norm(zs)
        if args.fid==True:

            y_hat= []

            for i in range(int(zs.shape[0]/args.test_size)):
                
                k= i*args.test_size

                if args.bench_mark:
                    y_temp= net.decoder(zs[k:k+args.test_size , :]).reconstruction.data.numpy()
                else:
                    y_temp= net.decode(zs[k:k+args.test_size , :]).data.numpy()

                y_hat.append(y_temp)
            
            y_hat = np.concatenate(y_hat, axis=0)

        else:
            if args.bench_mark:
                y_hat= net.decoder(zs).reconstruction.data.numpy()
            else:
                y_hat= net.decode(zs).data.numpy()

    elif args.task == "pca":
        z = []
        for yi, _ in test_loader:
            zi = net.encode(yi)
            z.append(zi.detach().numpy())
        z = np.concatenate(z, axis=0)

        pca = PCA(n_components=args.d)
        pca.fit(z)
        x = pca.transform(z)
        mu = np.average(x, axis=0)
        sigma = np.cov(x, rowvar=False)

        sigma_0 = np.sqrt(sigma[0][0])
        sigma_1 = np.sqrt(sigma[1][1])
        center = mu.copy()
        center[0] -= sigma_0 * 2
        center[1] -= sigma_1 * 2

        zs = []
        for i in range(args.X):
            tmp = []
            x = center.copy()
            x[0] += sigma_0 * i / args.X * 4
            for j in range(args.Y):
                x[1] += sigma_1 / args.Y * 4
                zi = pca.inverse_transform(x)
                tmp.append(zi)
            tmp = np.stack(tmp, axis=0)
            zs.append(tmp)
        zs = np.concatenate(zs, axis=0)
        zs = torch.Tensor(zs)

        y_hat = net.decode(zs).data.numpy()

    elif args.task=="svae":
        sampler = HypersphereUniformSampler(net)
        if args.fid:
            y_hat = (sampler.sample(num_samples = 10000))
        else:
            y_hat = (sampler.sample(num_samples = args.X*args.Y))
        y_hat = y_hat.cpu().data.numpy()
    elif args.task == "vamp":
        sampler = VAMPSampler(net)
        if args.fid:
            y_hat = (sampler.sample(num_samples = 10000))
        else:
            y_hat = (sampler.sample(num_samples = args.X*args.Y))
        y_hat = y_hat.cpu().data.numpy()

    elif args.task == "poisson":
        latent=1
        if args.dataset=='mnist':
            latent = 16
        elif args.dataset=="cifar10":
            latent = 128
        elif args.dataset=="celeba":
            latent = 64
        zs =  np.random.poisson(1.0,(args.X*args.Y,latent))
        # print(zs)
        zs = torch.Tensor(zs)
        # zs = zs - torch.mean(zs)
        # zs = norm(zs)
        y_hat= []
        for i in range(int(zs.shape[0]/args.test_size)):
            
            k= i*args.test_size

            if args.bench_mark:
                y_temp= net.decoder(zs[k:k+args.test_size , :]).reconstruction.data.numpy()
            else:
                y_temp= net.decode(zs[k:k+args.test_size , :]).data.numpy()

            y_hat.append(y_temp)
        
        y_hat = np.concatenate(y_hat, axis=0)
    elif args.task == "chisquare":
        latent=1
        if args.dataset=='mnist':
            latent = 16
        elif args.dataset=="cifar10":
            latent = 128
        elif args.dataset=="celeba":
            latent = 64
        zs =  np.random.chisquare(df=(latent-1),size=(args.X*args.Y,latent))
        # print(zs)
        zs = torch.Tensor(zs)
        zs = norm(zs)
        y_hat= []
        for i in range(int(zs.shape[0]/args.test_size)):
            
            k= i*args.test_size

            if args.bench_mark:
                y_temp= net.decoder(zs[k:k+args.test_size , :]).reconstruction.data.numpy()
            else:
                y_temp= net.decode(zs[k:k+args.test_size , :]).data.numpy()

            y_hat.append(y_temp)
        
        y_hat = np.concatenate(y_hat, axis=0)
        
    elif args.task == "uniform":
        latent=1
        if args.dataset=='mnist':
            latent = 16
        elif args.dataset=="cifar10":
            latent = 128
        elif args.dataset=="celeba":
            latent = 64
        zs =  np.random.uniform(low=0,high=1,size=(args.X*args.Y,latent))
        # zs = zs - np.mean(zs,axis=1)
        # print(zs)
        zs = torch.Tensor(zs)
        # zs = zs - torch.mean(zs)
        # zs = norm(zs)
        y_hat= []
        for i in range(int(zs.shape[0]/args.test_size)):
            
            k= i*args.test_size

            if args.bench_mark:
                y_temp= net.decoder(zs[k:k+args.test_size , :]).reconstruction.data.numpy()
            else:
                y_temp= net.decode(zs[k:k+args.test_size , :]).data.numpy()

            y_hat.append(y_temp)
        
        y_hat = np.concatenate(y_hat, axis=0)

    elif args.task == "plot":
        # Extract x and y components
        z = []
        i = 0
        for yi, _ in test_loader:
            # print(yi)
            if args.bench_mark:
                zi = net.encoder(yi).embedding
            else:
                zi = net.encode(yi)
            z.append(zi.detach().numpy())
            i += 1
            if i > 1 :
                break
        z = np.concatenate(z, axis=0)
        z = torch.Tensor(z)
        z = norm(z)
        z = z.numpy()
        x = [vector[0] for vector in z]
        y = [vector[1] for vector in z]

        # Create the plot
        plt.figure(figsize=(8, 8))

        # Scatter plot
        plt.scatter(x, y, color='orange', label='Data Points')

        # Line plot connecting the points
        # plt.plot(x, y, color='orange', linestyle='-', marker='o', label='Connecting Line')

        # Adding labels and title
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.title(f'Latent space of {args.model}')
        plt.legend()

        # Show the plot
        plt.savefig(f"plots/image_{args.model}_100.png",dpi=400)
        return
    if args.task == "tsne":
        z = []
        if args.model == 'svae':
            sampler = HypersphereUniformSampler(net)
            y_hat = sampler.sample(num_samples = 2000)
            y_hat = y_hat.cpu().data.numpy()
            for yi, _ in test_loader:
                yi = yi.cuda()
                if args.bench_mark:
                    zi = net.encoder(yi).embedding
                else:
                    zi = net.encode(yi)
                z.append(zi.cpu().detach().numpy())
            z = np.concatenate(z, axis=0)
            tsne = TSNE(n_components=2, random_state=42)
            tsne_points = tsne.fit_transform(z)
            tsne = TSNE(n_components=2, random_state=42)
            svae_points = tsne.fit_transform(y_hat)
            plt.figure(figsize=(10, 8))
            plt.scatter(tsne_points[:, 0], tsne_points[:, 1], c='blue', label='Original Images')
            plt.scatter(svae_points[:, 0], svae_points[:, 1], c='orange', label='Sampled Images from SVAE')
            plt.title('Original vs Sampled Images')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.legend()
            plt.savefig(f'results/mnist2k/tsne_vs_svae_points_{args.model}.png')
        elif args.model == "vamp":
            net = net.cpu()
            sampler = VAMPSampler(net)
            y_hat = (sampler.sample(num_samples = 2000))
            y_hat = y_hat.cpu().data.numpy()
            for yi, _ in test_loader:
                yi = yi.cuda()
                if args.bench_mark:
                    zi = net.encoder(yi).embedding
                else:
                    zi = net.encode(yi)
                z.append(zi.cpu().detach().numpy())
            z = np.concatenate(z, axis=0)
            tsne = TSNE(n_components=2, random_state=42)
            tsne_points = tsne.fit_transform(z)
            tsne = TSNE(n_components=2, random_state=42)
            vamp_points = tsne.fit_transform(y_hat)
            plt.figure(figsize=(10, 8))
            plt.scatter(tsne_points[:, 0], tsne_points[:, 1], c='blue', label='Original Images')
            plt.scatter(vamp_points[:, 0], vamp_points[:, 1], c='orange', label='Sampled Images from VAMP')
            plt.title('Original vs Sampled Images')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.legend()
            plt.savefig(f'results/mnist2k/tsne_vs_vamp_points_{args.model}.png')
        else:
            for yi, _ in test_loader:
                if args.bench_mark:
                    zi = net.encoder(yi).embedding
                else:
                    zi = net.encode(yi)
                z.append(zi.detach().numpy())
            z = np.concatenate(z, axis=0)
            gmm = mixture.GaussianMixture(
                n_components=args.d, covariance_type='full')
            mu = np.average(z, axis=0)
            sigma = np.cov(z, rowvar=False)
            gmm.fit(z)
            gmm_points = gmm.sample(2000)
            if args.model in ['vae','wae','betavae','hvae','vaeiaf']:
                mu = np.zeros((len(mu)), dtype=float)
                sigma = np.identity(len(sigma),dtype=float)
            mvg_points = np.random.multivariate_normal(mu, sigma, 2000)
            # print(mvg_points)
            data,label = gmm_points
            tsne = TSNE(n_components=2, random_state=42)
            gmm_points = tsne.fit_transform(data)
            tsne = TSNE(n_components=2, random_state=42)
            mvg_points = tsne.fit_transform(mvg_points)
            tsne = TSNE(n_components=2, random_state=42)
            tsne_points = tsne.fit_transform(z)
            # Plot 1: t-SNE points and GMM points
            plt.figure(figsize=(10, 8))
            plt.scatter(tsne_points[:, 0], tsne_points[:, 1], c='blue', label='Original Images')
            plt.scatter(gmm_points[:, 0], gmm_points[:, 1], c='orange', label='Sampled Images from GMM')
            plt.title('Original vs Sampled Images')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.legend()
            plt.savefig(f'results/mnist2k/tsne_vs_gmm_points_{args.model}.png')

            # Plot 2: t-SNE points and MVG points
            plt.figure(figsize=(10, 8))
            plt.scatter(tsne_points[:, 0], tsne_points[:, 1], c='blue', label='Original Images')
            plt.scatter(mvg_points[:, 0], mvg_points[:, 1], c='orange', label='Sampled Images from MVG')
            plt.title('Original vs Sampled Images')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.legend()
            plt.savefig(f'results/mnist2k/tsne_vs_mvg_points_{args.model}.png')
        return 


    elif args.task == "kde":
        # Extract x and y components
        z = []
        i = 0
        for yi, _ in test_loader:
            # print(yi)
            if args.bench_mark:
                zi = net.encoder(yi).embedding
            else:
                zi = net.encode(yi)
            z.append(zi.detach().numpy())
            # i += 1
            # if i > 1 :
            #     break
        z = np.concatenate(z, axis=0)
        z = torch.Tensor(z)
        z = norm(z)
        z = z.numpy()
        x = [vector[0] for vector in z]
        y = [vector[1] for vector in z]
        # xy = np.vstack([x, y])
        # kde = gaussian_kde(xy)
        # density = kde(xy)
        # # Create the plot
        # plt.figure(figsize=(10, 8))

        # # Scatter plot
        # plt.scatter(x, y, c=density, s=50, edgecolor='red')
        # plt.colorbar(label='Density')

        # # Line plot connecting the points
        # # plt.plot(x, y, color='orange', linestyle='-', marker='o', label='Connecting Line')

        # # Adding labels and title
        # plt.xlabel('X Axis')
        # plt.ylabel('Y Axis')
        # plt.title(f'Latent space of {args.model}')
        # # plt.legend()
        # Gaussian KDE
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = kde(positions).reshape(xx.shape)

        # Create the density plot
        plt.figure(figsize=(10, 8))

        # Contour plot
        plt.contourf(xx, yy, density, cmap='Blues')
        plt.colorbar(label='Density')
        # Show the plot
        plt.savefig(f"plots/image_{args.model}_kde_2.png",dpi=400)
        return
    if args.fid:
        print("Saving start")
        test_data_path= './test_image_folder/'+ args.dataset
        
        if not os.path.exists(test_data_path):
            os.makedirs(test_data_path)
            real_hat=[]
            
            if args.dataset== 'cifar10':
                # Define the number of images per class you want to use
                num_images_per_class = 1000 #500
                # Create a new dataset with a subset of images per class
                subset = []
                class_count = [0] * 10  # To keep track of the number of images per class

                for i in range(len(test_set)):
                    image, label = test_set[i]
                    if class_count[label] < num_images_per_class:
                        subset.append((image, label))
                        class_count[label] += 1
                    if sum(class_count) == 10 * num_images_per_class:
                        break  
                test_loader= torch.utils.data.DataLoader(subset,
                                                        num_workers=32,
                                                        batch_size=args.test_size
                                                        )

            for re, label in test_loader:
                img= re.numpy()
                real_hat.append(img)
            
            real_hat= np.concatenate(real_hat, 0)
            
            
            # Iterate over the stacked images and save each one individually
            for i, image in enumerate(real_hat):
                
                if args.dataset=='mnist' or args.dataset=='fmnist':
                    
                    # Reshape the image from (1, 28, 28) to (28, 28)
                    image = np.squeeze(image, axis=0)
                    # print(np.max(image),np.min(image))
                    # break
                    image[image>1]=1
                    image[image<0]=0
                    # Normalize the pixel values to the range of 0 to 1

                    # Create a PIL Image object from the numpy array
                    image = Image.fromarray((image * 255).astype(np.uint8))
                
                else:
                    # Reshape the image from (3, 64, 64) to (64, 64, 3)
                    image = np.transpose(image, (1, 2, 0))
                    #print(np.max(image), np.min(image))
                    
                    # Normalize the pixel values to the range of 0 to 1
                    # image = image.astype(np.float32) / 255.0

                    # Create a PIL Image object from the numpy array
                    image = Image.fromarray((image * 255).astype(np.uint8))

                # Save the image with a unique filename
                filename = f"image_{i+1}.png"
                filepath = os.path.join(test_data_path, filename)
                image.save(filepath)
                if i+1==10000:
                    break

            print("Test Images saved successfully.")
        
        
        if args.vae:
            gen_data_path= './generated_image_folder/'+ args.dataset + '/' + args.task + '/vae' + f"_{args.lr}"
        
        elif args.vanilla:
            gen_data_path= './generated_image_folder/'+ args.dataset + '/' + args.task + '/vanilla'  + '_n'+str(args.n)+ f"_{args.lr}"
        
        elif args.irmae:
            gen_data_path= './generated_image_folder/'+ args.dataset + '/' + args.task + '/irmae_l' + str(args.t)  + '_n' + str(args.n)+ f"_{args.lr}"
        elif args.bench_mark:
            gen_data_path= './generated_image_folder/'+ args.dataset + '/' + args.task + '/l' + str(args.t) + '_n' + str(args.n) + '_' + str(args.optimizer) + '_unif_lambda' + str(args.unif_lambda) + f"_{args.lr}_d_{args.d}_model_{args.model}" 
        else:
            gen_data_path= './generated_image_folder/'+ args.dataset + '/' + args.model + '/' + args.task + '/l' + str(args.t) + '_n' + str(args.n) + '_' + str(args.optimizer) + '_unif_lambda' + str(args.unif_lambda) + f"_{args.lr}_d_{args.d}" 
        
        if not os.path.exists(gen_data_path):
            os.makedirs(gen_data_path)
            
            # Iterate over the stacked images and save each one individually
            for i, image in enumerate(y_hat):
                
                if args.dataset=='mnist' or args.dataset=='fmnist':

                    # Reshape the image from (1, 28, 28) to (28, 28)
                    image = np.squeeze(image, axis=0)
                    image[image>1]=1
                    image[image<0]=0

                    # Create a PIL Image object from the numpy array
                    image = Image.fromarray((image * 255).astype(np.uint8))

                    # Create a PIL Image object from the numpy array

                
                else:
                    # Reshape the image from (3, 64, 64) to (64, 64, 3)
                    image = np.transpose(image, (1, 2, 0))

                    # Create a PIL Image object from the numpy array
                    image = Image.fromarray((image * 255).astype(np.uint8))

                # Save the image with a unique filename
                filename = f"image_{i+1}.png"
                filepath = os.path.join(gen_data_path, filename)
                image.save(filepath)

            print("Generated Images saved successfully.")
        
        else:
            print('Gen img folder already exist, delete and re-run')
      
    
    else:
        if args.X == 1:
            axs = np.expand_dims(axs, axis=0)
        if args.Y == 1:
            axs = np.expand_dims(axs, axis=1)
        if not args.org:
            # if args.task != "interpolation":
            #     y_hat = y_hat[args.batch*args.X*args.Y:]
            # y_hat = y_hat[args.batch*args.X*args.Y:]
            for i in range(args.X):
                for j in range(args.Y):
                    if args.dataset == 'mnist' or args.dataset== 'fmnist':
                        im = y_hat[i*args.Y+j][0, :, :]
                    else:
                        im = np.transpose(y_hat[i*args.Y+j], [1, 2, 0])
                    if args.dataset == 'mnist' or args.dataset=='fmnist':
                        axs[i, j].imshow(1-im, interpolation='nearest', cmap='Greys')
                    else:
                        axs[i, j].imshow(im, interpolation='nearest')
                    axs[i, j].axis('off')

            fig.tight_layout(pad=0.1)
            path = args.save_path
            model_name = f"{args.model}/"
            model_name = model_name.capitalize()
            path += model_name
            # if args.task=="reconstruction":
            #     path += "reconstruction/"
            # elif args.task=="mvg" or args.task=="stdn" or args.task=='gmm':
            #     path += "generation/"
            # elif args.task=="interpolation":
            #     path += "interpolation/"
            path += f"{args.dataset}/"
            if not os.path.exists(path):
                os.makedirs(path)
            print(path)
            plt.savefig(path+f"{args.task}_{args.model}_{args.d}.png",dpi=5000)
        else:
            # if args.task != "interpolation":
            #     y_hat = y_hat[args.batch*args.X*args.Y:]
            for i in range(args.X):
                for j in range(args.Y):
                    if args.dataset == 'mnist' or args.dataset== 'fmnist':
                        im = y_org[i*args.Y+j][0, :, :]
                    else:
                        im = np.transpose(y_org[i*args.Y+j], [1, 2, 0])
                    if args.dataset == 'mnist' or args.dataset=='fmnist':
                        axs[i, j].imshow(1-im, interpolation='nearest', cmap='Greys')
                    else:
                        axs[i, j].imshow(im, interpolation='nearest')
                    axs[i, j].axis('off')

            fig.tight_layout(pad=0.1)
            path = f"results/"
            model_name = f"Org/"
            model_name = model_name.capitalize()
            path += model_name
            if args.task=="reconstruction":
                path += "reconstruction/"
            elif args.task=="mvg" or args.task=="stdn" or args.task=='gmm':
                path += "generation/"
            elif args.task=="interpolation":
                path += "interpolation/"
            path += f"{args.dataset}/{args.task}/"
            if not os.path.exists(path):
                os.makedirs(path)
            
            plt.savefig(path+f"{args.task}_org.png",dpi=5000)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
