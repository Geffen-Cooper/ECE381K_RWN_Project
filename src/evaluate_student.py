import dgl
import torch
import dgl.data
import networkx as nx
import metis
import matplotlib.pyplot as plt
import numpy as np
from models import *
from train import *

models = ["GCN"]#,"GAT","GSAGE"]
ks = [1,2,5,10,20]
datasets = ["cora"] #,"citeseer"]#,"arxiv"]


# iterate over the datasets
for dataset in datasets:
    # first load the dataset, split into k partitions
    # dataset_nx, dataset_dgl, data = load_dataset(dataset)

    
    # model_accs = []
    plt.figure()
    # iterate over the models
    for idx,model in enumerate(models):
        teacher_partition_accs = []
        student_partition_accs = []
        studentbig_partition_accs = []
        #student1L_partition_accs = []
        # iterate over the possible ks
        for k in ks:
            # if k == 1:
            #     partitions = [dataset_dgl]
            # else:
            #     partitions, parts_tensor = partition_network(k, dataset_nx, dataset_dgl)
            teacher_avg_acc = 0
            student_avg_acc = 0
            studentbig_avg_acc = 0
            #student1L_avg_acc = 0
            # avg over each partition for a given k
            for partition in range(k):
                teacher_checkpoint_path = "saved_models/best_"+str(model)+"_"+str(dataset)+"_"+"p"+str(partition+1)+"_k"+str(k)+".pth"
                student_checkpoint_path = "saved_models/best_student_validation"+str(model)+"_"+str(dataset)+"_"+"p"+str(partition+1)+"_k"+str(k)+".pth"
                studentbig_checkpoint_path = "saved_models/best_student_validationbig"+str(model)+"_"+str(dataset)+"_"+"p"+str(partition+1)+"_k"+str(k)+".pth"
                #student1L_checkpoint_path = "saved_models/best_student_validation1L"+str(model)+"_"+str(dataset)+"_"+"p"+str(partition+1)+"_k"+str(k)+".pth"
                
                teacher_checkpoint = torch.load(teacher_checkpoint_path)
                student_checkpoint = torch.load(student_checkpoint_path)
                studentbig_checkpoint = torch.load(studentbig_checkpoint_path)
                #student1L_checkpoint = torch.load(student1L_checkpoint_path)

                # create a gnn for this partition using graph parameters
                # features = partitions[partition].ndata['feat']
                # labels = partitions[partition].ndata['label']
                # test_mask = partitions[partition].ndata['test_mask']
                # m = load_model(model,features,data.num_classes, 3)
                # m.load_state_dict(checkpoint['model_state_dict'])

                # with torch.no_grad():
                #     # Forward
                #     logits = m(partitions[partition], features)
                #     pred = logits.argmax(1)

                #     # Compute loss and accuracy
                #     test_loss = F.cross_entropy(logits[test_mask], labels[test_mask])
                #     test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

                # avg_acc += test_acc

                # sum up the val accuracies normalized by total validation set size
                #print(checkpoint['val_acc'],checkpoint['val_size'].cpu(),checkpoint['total_val_size'])
                teacher_avg_acc += (teacher_checkpoint['val_acc']*(teacher_checkpoint['val_size'].cpu()/teacher_checkpoint['total_val_size']))
                student_avg_acc += (student_checkpoint['val_acc']*(student_checkpoint['val_size'].cpu()/student_checkpoint['total_val_size']))
                studentbig_avg_acc += (studentbig_checkpoint['val_acc']*(studentbig_checkpoint['val_size'].cpu()/studentbig_checkpoint['total_val_size']))
                #student1L_avg_acc += (student1L_checkpoint['val_acc']*(student1L_checkpoint['val_size'].cpu()/student1L_checkpoint['total_val_size']))
            #avg_acc /= k
            # avg acc for a given partition, should have len(ks) of these
            teacher_partition_accs.append(teacher_avg_acc)
            student_partition_accs.append(student_avg_acc)
            studentbig_partition_accs.append(studentbig_avg_acc)
            #student1L_partition_accs.append(student1L_avg_acc)
        # model_accs.append(partition_accs)
        plt.plot(ks,teacher_partition_accs)
        plt.plot(ks,studentbig_partition_accs)
        plt.plot(ks,student_partition_accs)
        #plt.plot(ks,student1L_partition_accs)
        
        plt.suptitle('Plot of accuracy versus k-way partitions: '+dataset, fontsize=12)
        plt.xlabel('K partitions', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.yticks(np.arange(0.5, 1, 0.1))
        plt.xticks([1,2,5,10,20])
    plt.legend(['teacher','student_big','student'])
plt.show()
