import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
class Visualization(object):
    def __init__(self, enable, max):
        self.enable = enable
        if self.enable:
            plt.ion()
            self.figure = plt.figure(num='TPJ', frameon=True)
            self.figure.canvas.mpl_connect('close_event', self.__class__.__exit)
            self.axes3d = self.figure.add_subplot(projection='3d')  #gca   
            plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
            if max: plt.get_current_fig_manager().window.state('zoomed')   #plt.get_backend() ==  'TkAgg'
            self.god_X = None
            self.god_Y = None
            self.god_Z = None
            self.fit_X = None
            self.fit_Y = None
            self.fit_Z = None
            self.run_xyz = {}
            self.loss = None

    def __exit(event): 
        import os; os._exit(0)

    def set_god_surface(self, X, Y, Z):
        if self.enable:
            self.god_X = X
            self.god_Y = Y
            self.god_Z = Z
            self.__show_all()

    def set_fit_surface(self, X, Y, Z):
        if self.enable:
            self.fit_X = X
            self.fit_Y = Y
            self.fit_Z = Z
            self.__show_all()

    def set_fit_loss(self, l):
        if self.enable:
            self.loss = l
            self.__show_all()

    def add_run_trajectory(self, x, y, z, e):
        if self.enable:
            if e not in self.run_xyz:
                self.run_xyz[e] = ([],[],[])
            self.run_xyz[e][0].append(x)
            self.run_xyz[e][1].append(y)
            self.run_xyz[e][2].append(z)
            self.__show_all()

    def __show_all(self):
        if self.enable:
            self.axes3d.clear()
            if self.god_X is not None and self.god_Y is not None and self.god_Z is not None:
                self.axes3d.contour(self.god_X, self.god_Y, self.god_Z, zdir='z', offset=np.min(self.god_Z), colors='gray', extend3d=False) 
                self.axes3d.plot_surface(self.god_X, self.god_Y, self.god_Z, rstride=1, cstride=1, cmap='rainbow', alpha=0.5, linewidth=3)
            if self.fit_X is not None and self.fit_Y is not None and self.fit_Z is not None:
                self.axes3d.plot_surface(self.fit_X, self.fit_Y, self.fit_Z, rstride=1, cstride=1, cmap='coolwarm', alpha=0.8, linewidth=1)    
                self.axes3d.scatter(self.fit_X, self.fit_Y, self.fit_Z, zdir='z', s=3, c='green', depthshade=True)        
            if self.loss is not None:
                self.axes3d.text(0, 0, np.max(self.god_Z), str(self.loss), bbox=dict(facecolor='green', alpha=0.66))
            if self.run_xyz is not None:
                colors = ['red','green','blue']
                for e,(run_x, run_y, run_z) in self.run_xyz.items():
                    self.axes3d.plot(run_x, run_y, run_z, color=colors[e%len(colors)], linewidth=1)
                    self.axes3d.scatter(run_x, run_y, run_z, color=colors[e%len(colors)], s=11, marker='*')
            self.__wait(0.001)

    def __wait(self, time):
        if self.enable:
            try:
                plt.pause(time)
            except Exception as e:
                pass

    def save(self, file_name):
        if self.enable:
            if file_name is not None:
                self.figure.savefig(file_name)

    def show(self):
        if self.enable:
            plt.ioff()
            plt.show()

import numpy as np
class World(object):
    def __init__(self, mode):
        self.mode = mode

    def make(self):   
        self.random = 0   
        if self.mode == 'function-fitting-2d': 
            x = np.linspace(-0.99, +0.99, 15)
            y = np.linspace(-0.01, +0.01, 15) 
        elif self.mode == 'reinforcement-learning':
            x = np.linspace(-0.63, +0.63, 15)
            y = np.linspace(-0.63, +0.63, 15)  
        else:
            raise     

        if self.mode == 'function-fitting-2d':
            self.god_x, self.god_y = np.meshgrid(x, y)  
            self.god_z = self._fit(self.god_x) + self._fit(self.god_y)
        elif self.mode == 'reinforcement-learning':
            self.god_x, self.god_y = np.meshgrid(x, y)  
            self.god_z = self._map(self.god_x) + self._map(self.god_y)
        else:
            raise

    def take(self, old_state, action):
        x = old_state[0] + action[0]
        y = old_state[1] + action[1]
        if self.mode == 'function-fitting-2d':
            z = self._fit(x) + self._fit(y)
            return z
        elif self.mode == 'reinforcement-learning':
            z = self._map(x) + self._map(y)
            new_state = [x, y]
            z = self._map(x) + self._map(y)
            reward = self._z2r(z)
            return new_state, reward, z
        else:
            raise

    def noise(self, flag):
        self.random = flag
        print('World.noise, random=',self.random)

    def _fit(self, i):
        #o = i                                                   #identity
        o = np.tanh(9*i)                                         #tanh
        #o = np.where(i>=0,0.5,0)                                #step: must be 0.5, due to the x+y
        #o = np.sin(3*i)                                         #sin: if coefficient is too large, can not fit, need linear and layer  
        #o = np.log(abs(i)+0.00000001)                           #log, need better learning-rate and early-stop point, due to variable valid range, so abs and add    
        #o = np.power(0.7*i, 3)                                  #pow+odd
        #o = np.power(0.7*i, 2)                                  #pow+even: DIFFICULT, Done: MultiLayer(Linear+Universal)  Linear(more nodes)
        #o = np.sin(3*i)*np.random.binomial(n=1,p=0.5,size=1)    #TODO
        return o

    def _map(self, i):
        o = self.__base(i) #+ self.__period(i) + self.__trend(i) + self.__noise(i)
        return o

    def __base(self, i):
        o = np.power((i*2),4) - np.power((i*1),3) - np.power((i*2),2)
        return o

    def __period(self, i):  #TODO: time-base-multiple-values-function, todo hard-multiple-values-function
        o = np.sin(i) / 0.5
        return o

    def __trend(self, i):
        o = np.log(i+1) / 2 
        return o

    def __noise(self, i):
        o = np.random.rand(1) * 0.01 if self.random else 0
        return o

    def _z2r(self, z):
        if z < -0.3:  #hidden-unknow
            reward = -0.9
        elif z > +0.3:
            reward = +0.9
        else:
            reward = 0.0
        return reward

import torch
class Universal(torch.nn.Module):
    def __init__(self, weight_size):
        super(Universal, self).__init__()
        self.A = torch.nn.parameter.Parameter(torch.Tensor(1, weight_size))   #1 or batch_size?  #boardcast?  #!!! 1 is enought
        self.B = torch.nn.parameter.Parameter(torch.Tensor(1, weight_size))  
        self.C = torch.nn.parameter.Parameter(torch.Tensor(1, weight_size))  
        #self.D = torch.nn.parameter.Parameter(torch.Tensor(1, weight_size)) 
        self.reset_parameters()       
        
    def reset_parameters(self):
        print('Universal.reset_parameters()')
        torch.nn.init.ones_(self.A)   #torch.nn.init.uniform_(self.A, 0.9, 1.1)
        torch.nn.init.ones_(self.B)
        torch.nn.init.ones_(self.C)
        #torch.nn.init.ones_(self.D)

    def forward(self, X):  # AX/(abs(BX)+C)+D
        AX = self.A.mul(X)  #torch.mul, same as *, hadamard product, one of four product operations
        BX = self.B.mul(X)
        ABS = abs(BX)
        ADD = ABS + self.C
        DIV = AX / ADD
        #Y = DIV + D
        return DIV

class Piecewise(torch.nn.Module):
    def __init__(self, weight_size_a, weight_size_b):
        super(Piecewise, self).__init__()
        self.A = torch.nn.parameter.Parameter(torch.Tensor(weight_size_a, weight_size_b))
        self.B = torch.nn.parameter.Parameter(torch.Tensor(weight_size_a, weight_size_b))  
        self.reset_parameters()       
        
    def reset_parameters(self):
        torch.nn.init.normal_(self.A, mean=-0.5, std=1.0)
        torch.nn.init.normal_(self.B, mean=+0.5, std=1.0)

    def forward(self, A, B):
        AX = A.matmul(self.A)
        BX = B.matmul(self.B)
        ADD = AX + BX
        return ADD

import torch
class Mirror(torch.nn.Module):
    def __tpj_disable_enable_requires_grad(m):
        print('disable_enable_requires_grad:  m.__class__.__name__=',m.__class__.__name__)
        if m.__class__.__name__.find('Universal') != -1:
            m.A.requires_grad = True
            m.B.requires_grad = True
            m.C.requires_grad = True
        elif m.__class__.__name__.find('Piecewise') != -1:
            m.A.requires_grad = True
            m.B.requires_grad = True
        elif m.__class__.__name__.find('Linear') != -1:
            if hasattr(m.bias, 'data'):
                m.weight.requires_grad = True
                m.bias.requires_grad = True
                #torch.nn.init.constant_(m.weight.data, val=1.0)
            else:
                m.weight.requires_grad = True 
                #torch.nn.init.constant_(m.weight.data, val=1.0)           
        elif m.__class__.__name__ in ['Sequential','Mirror','Tanh']:
            print('m.__class__.__name__=',m.__class__.__name__,', skip this module and keep requires_grad.')
            pass
        else:
            print('m.__class__.__name__=',m.__class__.__name__,', check this module and set requires_grad.')
            raise
        print('\n')

    def __tpj_show_requires_grad(self):
        for m in self.parameters(): 
            print('show_requires_grad:', m.__class__.__name__, m.requires_grad)
        print('\n')

    def __tpj_show_requires_grad_and_parameter(self):
        for m in self.parameters():
            print('show_requires_grad_and_parameter:',m.__class__.__name__, m.requires_grad, m, '\n')
        print('\n')

    def __init__(self, i_size, o_size, batch_size, combination):
        super().__init__()
        self.combination = combination
        if self.combination == "Universal":   #TPJ
            h_size = i_size * 1  #MUST be 1
            self.mapping = Universal(h_size)
        elif self.combination == "Piecewise":   #A,B,...,N?
            h_size = i_size * 1  #MUST be 1
            self.universalA = Universal(h_size)
            self.universalB = Universal(h_size)
            self.piecewise = Piecewise(h_size, h_size)
        elif self.combination == "Linear+Universal":
            h_size = i_size * 1
            body = []
            body.append(torch.nn.Linear(h_size, h_size, bias=False))
            body.append(Universal(h_size))
            self.mapping = torch.nn.Sequential(*(body))
        elif self.combination == "MultiLayer(Linear+Universal)":   #TPJ
            h_size = i_size * 9   #if pow, must greater    
            head = [torch.nn.Linear(i_size, h_size)]
            body = []
            for i in range(1):
                if i==0:
                    body.append(torch.nn.Tanh())
                body.append(torch.nn.Linear(h_size, h_size))
                body.append(Universal(h_size))
            tail = [torch.nn.Linear(h_size, o_size)]
            self.mapping = torch.nn.Sequential(*(head+body+tail))
        elif self.combination == "Linear+Tanh":
            h_size = i_size * 1
            body = []
            body.append(torch.nn.Linear(h_size, h_size, bias=False))
            body.append(torch.nn.Tanh())
            self.mapping = torch.nn.Sequential(*(body))
        elif self.combination == "MultiLayer(Linear+Tanh)": 
            h_size = i_size * 9   #if pow, must greater    
            head = [torch.nn.Linear(i_size, h_size)]
            body = []
            for i in range(1):
                if i==0:
                    body.append(torch.nn.Tanh())
                body.append(torch.nn.Linear(h_size, h_size))
                body.append(torch.nn.Tanh())
            tail = [torch.nn.Linear(h_size, o_size)]
            self.mapping = torch.nn.Sequential(*(head+body+tail))
        else:
            raise

        self.apply(self.__class__.__weights_init)
        self.__tpj_show_requires_grad()   #TPJ

    def forward(self, x, debug=0, epoch=-1, peroid=100):
        if self.combination == "Piecewise":
            A = self.universalA.forward(x)
            B = self.universalB.forward(x)
            o = self.piecewise.forward(A, B)
        else:
            o = self.mapping.forward(x)

        if debug and epoch%peroid==0:
            self.__tpj_show_requires_grad_and_parameter()   #TPJ

        return o

    def __weights_init(m):
        if m.__class__.__name__.find('Linear') != -1:
            torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.25)
            if hasattr(m.bias, 'data'):     
                torch.nn.init.constant_(m.bias.data, val=0.0)
        Mirror.__tpj_disable_enable_requires_grad(m)   #TPJ

class TPJ(object):
    def tpj(world, visualization):
        device = ['cpu','cuda'][torch.cuda.is_available()]
        if world.mode == 'function-fitting-2d':
            i_size = 2  #x   y
            o_size = 1  #z
            combination = "Universal"
        elif world.mode == 'reinforcement-learning':
            i_size = 4  #state-x:1   state-y:1   action:2
            o_size = 3  #state-x:1   state-y:1   reward:1
            combination = "MultiLayer(Linear+Tanh)"
        else:
            raise
        batch_size = 3
        around_size = 1
        mirror = Mirror(i_size=i_size, o_size=o_size, batch_size=batch_size*around_size, combination=combination).to(device)
        optimizer = torch.optim.SGD(mirror.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss().to(device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.9, patience=100*10, verbose=False) 

        import random
        for e in range(100*300):
            i = []
            t = []
            for b in range(batch_size):
                random_landing_x = random.randint(0, len(world.god_x[0])-1)
                random_landing_y = random.randint(0, len(world.god_y)-1)
                if world.mode == 'function-fitting-2d':
                    old_state = [world.god_x[0][random_landing_x], world.god_y[random_landing_y][0]]
                    for f in range(around_size):
                        now_action = [0, 0]
                        new_z = world.take(old_state, now_action)
                        i.append(old_state)
                        t.append(new_z)
                elif world.mode == 'reinforcement-learning':
                    old_state = [world.god_x[0][random_landing_x], world.god_y[random_landing_y][0]]
                    for f in range(around_size):
                        scale = 2
                        actions = [[+0.1/scale,+0.1/scale],[-0.1/scale,-0.1/scale],[+0.1/scale,-0.1/scale],[-0.1/scale,+0.1/scale]]
                        now_action = actions[random.randint(0, len(actions)-1)]
                        new_state, new_reward, _ = world.take(old_state, now_action)
                        i.append(old_state+now_action)
                        t.append(new_state+[new_reward])
                        old_state = new_state
                else:
                    pass
            i = np.array(i)
            t = np.array(t)
            I = torch.autograd.Variable(torch.from_numpy(i.reshape(-1, i_size)).float(), requires_grad=False).to(device)
            O = mirror.forward(I, debug=0, epoch=e, peroid=100)
            T = torch.autograd.Variable(torch.from_numpy(t.reshape(-1, o_size)).float(), requires_grad=False).to(device)
            loss = criterion(O, T)            
            optimizer.zero_grad()
            loss.backward()        
            optimizer.step()
            scheduler.step(loss)
            if e % 100 == 0:
                #if loss.detach().cpu().numpy() < 0.05: world.noise(flag=0)           
                print('learn: loss=',loss.detach().cpu().numpy())
                if 0:
                    print('    T: ', end='')
                    for one in T.detach().cpu().numpy()[0]:
                        print('  %+.4f  '%one, end='')
                    print()
                    print('    O: ', end='')       
                    for one in O.detach().cpu().numpy()[0]:
                        print('  %+.4f  '%one, end='')
                    print()
            if world.mode == 'function-fitting-2d':
                if e % 100 == 0:
                    XG = world.god_x
                    YG = world.god_y
                    zs = [([0.0] * YG.shape[0]) for i in range(XG.shape[1])]
                    for ii in range(XG.shape[1]):
                        for jj in range(YG.shape[0]):
                            xy = [XG[0][ii], YG[jj][0]]
                            xy = np.array(xy)
                            I = torch.autograd.Variable(torch.from_numpy(xy.reshape(-1, i_size)).float(), requires_grad=False).to(device)
                            O = mirror.forward(I, debug=0)   
                            zs[jj][ii] = np.sum(O.detach().cpu().numpy()[0])  #TPJ  
                            #print('I=',I,'O=',O,'Z=',zs[jj][ii])
                    ZF = np.array(zs)        
                    visualization.set_fit_surface(XG, YG, ZF)
                    
            else:
                pass

def tpj():
    world = World(mode='function-fitting-2d')
    world.make()

    visualization = Visualization(enable=1, max=0)
    visualization.set_god_surface(world.god_x, world.god_y, world.god_z)

    TPJ.tpj(world, visualization)

    visualization.save(file_name='tpj.png')
    visualization.show()

if __name__ == '__main__':
    tpj()

