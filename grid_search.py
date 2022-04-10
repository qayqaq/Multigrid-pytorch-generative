import os

batchsize_list = [100]
learning_rate_list = [0.0001, 0.001, 0.01]
beta1_list = [0.5, 0.7, 0.9]
weight_decay_list = [0.00001, 0.0001, 0.001]
Langevin_T_list = [50, 100, 150]
delta_list = [0.0001, 0.001, 0.01]

n = 0
for batchsize in batchsize_list:
    for learning_rate in learning_rate_list:
        for beta1 in beta1_list:
            for weight_decay in weight_decay_list:
                for Langevin_T in Langevin_T_list:
                    for delta in delta_list:
                        command = "python main.py"     + \
                                  " --batchsize "      + str(batchsize)     + \
                                  " --learning_rate "  + str(learning_rate)          + \
                                  " --beta1 "          + str(beta1)    + \
                                  " --weight_decay "   + str(weight_decay) + \
                                  " --Langevin_T "     + str(Langevin_T)             + \
                                  " --delta "          + str(delta)
                        print(n, command)
                        n = n + 1
                        os.system(command)
