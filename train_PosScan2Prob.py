import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, TensorDataset
import matplotlib.pyplot as plt
import sys
import copy
import numpy as np
import random
import yaml
import matplotlib.pyplot as plt
import math




##### SET UP Train and Test function with classes for FC And CNN #####
def test(net, loader, device, batch_size=100, t=.5):
    # prepare model for testing (only important for dropout, batch norm, etc.)
    net.eval()
    
    test_loss = 0
    total=0
    samples_seen = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            
            # prep data
            data, target = data.to(device), target.to(device)
            data = data.float()
            
            # compute floss
            output = net(data)
            # output is (batchsizex21) tensor
            pred = torch.where(output>=t, 1.0, 0.0)
        
            correct += torch.sum(torch.all(torch.eq(pred,target), dim=1).float()).item()
            test_loss += F.binary_cross_entropy(output, target, reduction='sum')
            
            # update counter for batches
            total = total + 1
            samples_seen += data.shape[0]

    average_loss = test_loss/(len(loader.dataset))
    accuracy = 100*(correct/(samples_seen))
    print('Whole Scan Accuracy: {:.2f} %'.format(accuracy))
    
    return accuracy

def train(net, loader, optimizer, epoch, device, log_interval=50, t=.1, loss_type='BCE', class1_weight=1):
    # prepare model for training (only important for dropout, batch norm, etc.)
    net.train()
    train_loss = 0
    samples_seen = 0
    correct = 0
    correct0 = 0
    correct1 = 0
    num0_seen = 0
    num1_seen = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        # convert to float and send to device
        target = target.float()  
        data = data.float()
        data, target = data.to(device), target.to(device)
        
        # clear up gradients for backprop
        optimizer.zero_grad()
        output = net(data)

        # output is (batchsizex21) tensor
        pred = torch.where(output>=t, 1.0, 0.0)
        
        # compute 0 correct vs 1 correct

        inds0 = torch.where(target == 0.0)
        inds1 = torch.where(target == 1.0)
        num0 = len(inds0[0])
        num1 = len(inds1[0])

        num0_seen += num0
        num1_seen += num1
        c0 = torch.sum(torch.eq(pred[inds0], target[inds0]).float())
        c1 = torch.sum(torch.eq(pred[inds1], target[inds1]).float())
        correct0 += c0
        correct1 += c1
        correct += c0+c1
        '''
        if num0 == 0:
            w0 = 0
            w1 = 1
        elif num1 == 0:
            w0=1
            w1=0
        else:
            w0 = (num0+num1)/num0
            w1 = (num0+num1)/num1
        '''
        w0=1
        w1=class1_weight
        if loss_type == 'BCE':
            loss_func = F.binary_cross_entropy
        if loss_type == 'MSE':
            loss_func = F.mse_loss


        loss = torch.sum(loss_func(output[inds0], target[inds0], reduction='none')*(w0)) + \
              torch.sum(loss_func(output[inds1], target[inds1], reduction='none')*(w1))
        
        train_loss += loss
        samples_seen += data.shape[0]


        # compute gradients and make updates
        loss.backward()
        optimizer.step()
        
    accuracy = 100*(correct/(21*samples_seen))
    accuracy0 = 100*(correct0/(num0_seen))
    accuracy1 = 100*(correct1/(num1_seen))

    avg_loss = train_loss/len(loader.dataset)
    if epoch % log_interval == 0:
        print('\nTrain Epoch: {} Loss: {:.4f}   Accuracy:  {:.2f} \n\tAccuracy 0: {:.2f}, Accuracy 1: {:.2f}'\
              .format(epoch, avg_loss, accuracy, accuracy0, accuracy1))

    return accuracy, accuracy0, accuracy1, avg_loss


def custom_predict_yaml(model, inputs):
    weights = {}
    offsets = {}

    layerCount = 1
    activations = []

    for layer in range(1, len(model['weights']) + 1):
        
        weights[layer] = np.array(model['weights'][layer])
        offsets[layer] = np.array(model['offsets'][layer])

        layerCount += 1
        activations.append(model['activations'][layer])

    curNeurons = inputs

    for layer in range(layerCount-1):

        curNeurons = curNeurons.dot(weights[layer + 1].T) + offsets[layer + 1]

        if 'Sigmoid' in activations[layer]:
            curNeurons = 1/(1 + np.exp(-curNeurons))
        elif 'Tanh' in activations[layer]:
            curNeurons = np.tanh(curNeurons)

    return np.array(curNeurons)


def knockout(scan, probs, thresh=0.5):
    noised_scan = np.copy(scan)
    noised_scan[np.where(probs>=thresh)] = .5
    return noised_scan

def control_error_test(controller_filename, t, train_pos, train_clean_scans, real_controls, controller, network):

    
    gen_cont = np.zeros(len(train_pos))
    
    pos_tensor = torch.Tensor(s_train)
    i_output=network(pos_tensor).detach().numpy()
    control_error = 0
    
    for i in range(len(s_train)):

        clean_scan = np.copy(train_clean_scans[i])
        probs = np.copy(i_output[i])

        noised_scan2 = knockout(clean_scan, probs, thresh=t)

        gen_cont[i] =  custom_predict_yaml(controller, noised_scan2)

        
        control_error += np.abs(real_controls[i]-gen_cont[i])
    print("Control Error {:.2f}".format(control_error))
    return control_error, gen_cont


class State2Lidar(nn.Module):
    
    """
    The network will take a 3 element state var and produce a clear lidar scan. 
    """
    def __init__(self, in_dim, out_dim, num_hidden_layers, layer_sizes):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim        
        self.num_layers = 2*num_hidden_layers + 2
        self.layer_sizes = layer_sizes

        self.layer_list = nn.ModuleList()

        self.layer_list.append(nn.Linear(self.in_dim, self.layer_sizes[0]))
        self.layer_list.append(nn.Tanh())
        
        self.num_hidden_layers = num_hidden_layers
        
        # hidden layers (to be tanh'ed after)
        for i in range(self.num_hidden_layers-1):
            self.layer_list.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
            self.layer_list.append(nn.Tanh())

            

        self.layer_list.append(nn.Linear(self.layer_sizes[-1], self.out_dim))

        
    def forward(self, x):
        # apply non linearity on output
        for i in range(self.num_layers-1):
            x = self.layer_list[i](x)

        x = torch.sigmoid(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # this should print 'cuda' if you are assigned a GPU
    print(device)
    

    # Load data (input is position)
    s_train, i_train = np.load("Fixed_Clean_lidar_w_Pos_uncovered.npy"), np.load("Many_Hot_Knockout_Targets_t0.1242424.npy")
    real_scans = np.load("Fixed_Noisy_lidar_uncovered.npy")
    controller_real_outputs = np.load("ControllerOuputsOnRealData.npy")

    real_scans = np.load("Fixed_Noisy_lidar_uncovered.npy")
    types = ['DDPG', 'TD3']
    cs = ['1','2','3']
    szs = ['64x64','128x128']
    controllers = [] 
    for t in types:
        for s in szs:
            for c in cs:
                controllers.append('dnns/'+t+'_L21_'+s+'_C'+c+'.yml')


    #normalize the input state data [0, 1]
    for i in range(len(s_train)):
        s_train[i][-3] = (s_train[i][-3]+.75)/10
        s_train[i][-2] = (s_train[i][-2])/10
        s_train[i][-1] = (s_train[i][-1])/(np.pi*1.5)
    
    train_pos = s_train[:,-3:]
    train_clean_scans = s_train[:,0:-3]
    print(s_train.shape)
    print(i_train.shape)

    train_dataset = TensorDataset(torch.Tensor(s_train), torch.Tensor(i_train))
    # these are  hyper-parameters.
    lidar_size = 21
    input_dim = 3+lidar_size
    out_dim = lidar_size

    layer_widths = [100]
    layer_depths = [6]
    decays = [0.01, 0.1, .5]

    class1_ratios = [3,4]
    loss_types = ['BCE']
    model_number = 1
    for decay in decays:
        for layer_width in layer_widths:
            for layer_depth in layer_depths:
                for loss_type in loss_types:
                    for class1_ratio in class1_ratios:
                        print(decay, layer_width, layer_depth, loss_type, class1_ratio)

                        num_hidden_layers = layer_depth
                        layer_sizes = [layer_width for i in range(layer_depth)]


                        #build network
                        network = State2Lidar(in_dim=input_dim, out_dim=out_dim, num_hidden_layers=num_hidden_layers, layer_sizes=layer_sizes)
                        network.to(device)
                        weight_decay = decay

                        #
                        best_loss = 1000000
                        loss_delta = 1000000
                        epsilon = 0.0001
                        thresh= 0.5
                        losses = []
                        epoch = 0

        
                        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=25042, shuffle=True)
                        optimizer = optim.Adam(network.parameters(), lr=.001, weight_decay=weight_decay)
                        best_loss = 1000000
                        loss_delta = 1000000
                        epsilon = 0.0001
                        losses = []


                        while epoch <20000:
                            acc, acc0, acc1, avg_loss = train(network, train_loader, optimizer, epoch+1, device, log_interval=100, t=thresh, loss_type=loss_type, class1_weight=class1_ratio)
                            losses.append(avg_loss)
                            epoch += 1
                            if (epoch + 1) % 100==0:
                                test(network, train_loader, device, batch_size=25042, t=thresh)
                                #best_loss = min(losses)
                                loss_delta = best_loss- min(losses)
                                if loss_delta >0:
                                    best_loss = min(losses)
                        print("Training with batch size 25042, learning rate 0.001 complete: ")
                        test(network, train_loader, device, batch_size=25042, t=thresh)
                        print("Finding Optimal Tresholds and saving model")

                        network.cpu()
                        with torch.no_grad():
                            print()
                            weight_list = []
                            maxes = []
                            mins = []
                            for t,p in network.named_parameters():
                                if 'weight' in t:
                                    weight_list = list(np.array(p.detach()).flatten()) + weight_list
                                    print('Layer: ' + t)
                                    print('\tMax: ' + str(torch.max((p)))+'   MIN: ' + str(torch.min(p)))
                                    maxes.append(torch.max((p)))
                                    mins.append(torch.min(p))
                                    print('\tMean: ' + str(torch.mean(p)))
                                    print('\tVarience: ' + str(torch.var(p)))
                            print('\n'+'Overall Varience: ' +str(np.var(np.array(weight_list))))
                            print('Overall Mean: ' +str(np.mean(np.array(weight_list))))
                            print('Overall Min: ' + str(min(mins)) + '  Overall Max: ' + str(max(maxes)))
                            plt.boxplot(weight_list, vert=False)
                        #print('Test Loss: ' + str(test_loss.item()))
                        print(network.layer_list)
                        weights_mean = np.mean(np.array(weight_list))
                        weights_var = np.var(np.array(weight_list))
                        weights_min = min(mins)
                        weights_max = max(maxes)


                        ## we only care about exact match, so we need to compute 
                        num_t = 10
                        num_l = 21
                        opt_t = np.ones(num_l)*.5
                        threshes = np.linspace(0, 1, num=num_t)
                        s_tensor = torch.Tensor(s_train)
                        output = network(s_tensor)
                        target = torch.Tensor(i_train)
                        for i in range(5):
                            exact_matches = np.zeros(num_t)
                            for k in range(num_l):
                                exact_matches = np.zeros(num_t)
                                t_temp=torch.Tensor(np.copy(opt_t))
                                for t in range(len(threshes)):
                                    t_temp[k] = threshes[t]
                                    
                                    # output is (batchsizex21) tensor
                                    pred = torch.where(output>=t_temp, 1.0, 0.0)

                                    correct = torch.sum(torch.all(torch.eq(pred,target), dim=1).float()).item()
                                    #print(correct)
                                    exact_matches[t]=correct
                                    print(correct, t)

                                # now we checked for the whole thing, so update our optimal
                                opt_t[k] = threshes[np.argmax(exact_matches)]
                        print(opt_t)


                        
                                                # output is (batchsizex21) tensor
                        pred = torch.where(output>=torch.Tensor(opt_t), 1.0, 0.0)

                        correct = torch.sum(torch.all(torch.eq(pred,target), dim=1).float()).item()
                        whole_scan_acc = round(100*correct/len(s_train))
                        print(whole_scan_acc)
                        
                        PATH ='KnockoutNetworkAnalytics/Pos2Prob' + str(layer_depth) + 'x' + str(layer_width)+loss_type+'BalancedFixedStepDecay'+ str(weight_decay)+'WSA'+str(whole_scan_acc)+'Epochs'+str(epoch)+'WeightsMean'+str(weights_mean)[0:5]+'WeightsVar'+str(weights_var)[0:5]+'.pth'
                        torch.save(network.state_dict(), PATH)
                        #num_controllers = 12
                        #controller_errors = np.zeros(num_controllers)
                        #for c in range(num_controllers):
                        #    controller_filename = controllers[c
                        c=7
                        controller_filename = controllers[c]
                        with open(controller_filename, 'rb') as f:
                            controller = yaml.full_load(f)

                        real = controller_real_outputs[c,:]

                        gen_noisy = np.zeros(len(s_train))
                        s_tensor = torch.Tensor(s_train)
                        i_output=network(s_tensor).detach().numpy()
                        control_error = 0
                        for i in range(len(train_clean_scans)):

                            clean_scan = np.copy(train_clean_scans[i])
                            probs = np.copy(i_output[i])

                            noised_scan = knockout(clean_scan, probs, thresh=opt_t)
                            gen_noisy[i] =  custom_predict_yaml(controller, noised_scan)


                            control_error += np.abs(real[i]-gen_noisy[i])
                        #controller_errors[c] = control_error

                        PATH ='KnockoutNetworkAnalytics/PosScan2Prob'+loss_type+str(class1_ratio) + str(layer_depth) + 'x' + str(layer_width)+loss_type+'Decay'+ str(weight_decay)+'WSA'+str(whole_scan_acc)+'Control_error'+str(round(control_error))+'Epochs'+str(epoch)+'WeightsMean'+str(weights_mean)[0:5]+'WeightsVar'+str(weights_var)[0:5]+'.pth'
                        torch.save(network.state_dict(), PATH)
                        #np.save(PATH[0:-4]+'ControllerErrors.npy', controller_errors)
                        np.save(PATH[0:-4]+'OptimalThresholds.npy', opt_t)

                        output_activation = 'Sigmoid'
                        activs = []
                        for layer_name in network.layer_list: 
                            if  'Tanh' in str(layer_name) or 'Sigmoid' in str(layer_name):
                                activs.append(str(layer_name)[:-2])
                        activs.append(output_activation)

                        input_filename =  PATH
                        output_filename = PATH[0:-4]+'.yml'

                        model_dict = torch.load(input_filename)

                        dnn_dict = {}
                        dnn_dict['weights'] = {}
                        dnn_dict['offsets'] = {}
                        dnn_dict['activations'] = {}


                        total_layers = len(model_dict.keys())//2
                        layer_count = 1

                        for key in model_dict:
                            
                            # if it is a weight layer...
                            if 'weight' in key:
                                dnn_dict['weights'][layer_count] = []
                                for row in model_dict[key]:
                                    a = []
                                    for col in row: 
                                        a.append(float(col))
                                    dnn_dict['weights'][layer_count].append(a)
                            # if it is a bias layer... 
                            if 'bias' in key:
                                dnn_dict['offsets'][layer_count] = []
                                for row in model_dict[key]:
                                    dnn_dict['offsets'][layer_count].append(float(row))

                                
                                dnn_dict['activations'][layer_count] = activs[layer_count-1]
                                layer_count += 1

                        with open(output_filename, 'w') as f:
                            yaml.dump(dnn_dict, f)
                        print(output_filename)


