import os
import random
import json
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sentence_transformers import SentenceTransformer

import Contant as Con
from Model import Model
from Document import Document
from NN_model import Encoder
from loss import SupConLoss


def train(args, text, label_list, model, proto_labels=None, proto_center=None, flag=0):
    # Initialize optimizers for the backbone and projector
    optimizer_backbone = optim.Adam(model.backbone.parameters(), lr=args.lr_backbone)
    optimizer_head = optim.Adam(model.instance_projector.parameters(), lr=args.lr_head)

    L = torch.tensor(label_list).to(device)
    # If flag is 0, process labeled data; if flag is 1, process unlabeled data
    if flag == 0:
        proto_labels = torch.unique(L)

    criterion = SupConLoss(proto_labels, args.temperature).to(device)
    # Learning rate schedulers for both optimizers
    scheduler_backbone = StepLR(optimizer_backbone, step_size=10, gamma=0.1)
    scheduler_head = StepLR(optimizer_head, step_size=10, gamma=0.1)

    for epoch in range(args.epochs):
        optimizer_backbone.zero_grad()
        optimizer_head.zero_grad()
        Y = model(text)
        if flag == 0:
            proto_center = [Y[L == label].mean(dim=0).tolist() for label in proto_labels]

        loss = criterion(Y, L, proto_center).to(device)
        loss.backward()
        optimizer_backbone.step()
        optimizer_head.step()
        scheduler_backbone.step()
        scheduler_head.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {loss.item():.4f}")

def seed_torch(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)  # To ensure hash randomization is disabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Tweet',
                        choices=["Biomedical", "StackkOverflow", "News-T", "News-S", "News-TS", "Tweet"])
    parser.add_argument('--applyICF', type=bool, default=True)
    parser.add_argument('--applyBERT', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--feature_dim', type=int, default=128)
    parser.add_argument('--delta', type=float, default=0.44, help='similarity threshold')
    parser.add_argument('--alpha', type=float, default=0.06, help='')
    parser.add_argument('--beta', type=float, default=0.007, help='')
    parser.add_argument('--weight', type=float, default=0.12, help='weight of the two views')
    parser.add_argument('--temperature', type=float, default=0.17, help="temperature required by contrastive loss")
    parser.add_argument('--epsilon', type=float, default=0.75, help="threshold for merging two clusters")
    parser.add_argument('--time_elapsed', type=int, default=1200, help="maximum lifetime of a cluster")
    parser.add_argument('--updateNum', type=int, default=1500, help="buffer size, or the frequency of encoder updates")
    parser.add_argument('--seed', type=int, default=1, help="")
    parser.add_argument('--outputPath', type=str, default='results/')
    parser.add_argument('--datasetPath', type=str, default='data/')
    parser.add_argument('--lr_head', type=float, default=5e-4, help="")
    parser.add_argument('--lr_backbone', type=float, default=5e-6, help="")
    parser.add_argument('--label_rate', type=float, default=0.1, help="the percentage of labeled texts")
    args = parser.parse_args(argv)

    return args


if __name__ == "__main__":

    args = get_args(sys.argv[1:])
    seed_torch(args.seed)

    output = os.path.join(args.outputPath, f"{args.dataset}.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model = SentenceTransformer("all-mpnet-base-v1", device=device)

    list_of_text, list_of_label, list_of_objects, list_of_docId = [], [], [], []

    with open(os.path.join(args.datasetPath, args.dataset)) as input:
        for line in input:
            obj = json.loads(line.strip())
            list_of_objects.append(obj)
            list_of_text.append((obj["textCleaned"]))
            list_of_label.append(obj["clusterNo"])
            list_of_docId.append(obj["Id"])

    labeled_data_len = int(len(list_of_text) * args.label_rate)
    # Split the dataset into labeled and unlabeled data
    labeled_list_of_text = list_of_text[:labeled_data_len]
    labeled_list_of_label = list_of_label[:labeled_data_len]
    labeled_list_of_objects = list_of_objects[:labeled_data_len]
    labeled_list_of_docId = list_of_docId[:labeled_data_len]

    unlabeled_list_of_text = list_of_text[labeled_data_len:]
    unlabeled_list_of_objects = list_of_objects[labeled_data_len:]
    unlabeled_list_of_docId = list_of_docId[labeled_data_len:]

    label2id = {}

    dirichlet_model = Model(args)
    model = Encoder(text_model, args.feature_dim)
    model.to("cuda")
    # use labeled texts to initialize the encoder
    train(args, labeled_list_of_text, labeled_list_of_label, model, None, None, 0)
    print("----------------------------------")
    # deal labeled texts
    for obj, text, label in zip(labeled_list_of_objects, labeled_list_of_text,
                                labeled_list_of_label):

        doc_emb = model.forward_text(text)
        labeled_document = Document(obj, doc_emb, dirichlet_model.word_wid_map,
                                    dirichlet_model.wid_word_map,
                                    dirichlet_model.wid_docId, dirichlet_model.word_counter)
        if label in label2id:
            cluster_id = label2id[label]
            dirichlet_model.addDocumentIntoClusterFeature(labeled_document, cluster_id)
        else:
            dirichlet_model.createNewCluster(labeled_document)
            label2id[label] = dirichlet_model.cluster_counter[0]

    temp_label_list, temp_text_list = [], []

    for CF in dirichlet_model.active_clusters.values():
        CF[Con.I_cl] = 0
    time_stamp = 0

    # deal unlabeled texts, update the encoder
    for obj, text, doc_id in zip(unlabeled_list_of_objects, unlabeled_list_of_text,
                                 unlabeled_list_of_docId):

        # If the buffer is full
        if time_stamp != 0 and time_stamp % args.updateNum == 0:
            centers, p_labels = [], []
            for id in dirichlet_model.active_clusters:
                CF = dirichlet_model.active_clusters[id]
                p_labels.append(id)
                centers.append(CF[Con.I_cce])
            p_labels = torch.tensor(p_labels).to(device)
            centers = [tensor.squeeze().tolist() for tensor in centers]
            # retrain the encoder
            train(args, temp_text_list, temp_label_list, model, p_labels, centers, 1)
            print("----------------------------------")
            # clear the buffer
            temp_label_list, temp_text_list = [], []

        # deal the arriving text one by one
        doc_vec_new = model.forward_text(text)
        unlabeled_document = Document(obj, doc_vec_new, dirichlet_model.word_wid_map,
                                      dirichlet_model.wid_word_map,
                                      dirichlet_model.wid_docId,
                                      dirichlet_model.word_counter)

        dirichlet_model.sampleCluster(unlabeled_document)
        # add the unlabeled text and its pseudo-label into the buffer
        temp_text_list.append(text)
        temp_label_list.append(dirichlet_model.docIdClusId[doc_id])
        time_stamp += 1

    # save the result
    if not os.path.exists(args.outputPath):
        os.makedirs(args.outputPath)
    f = open(output, "w+")
    for d in dirichlet_model.docIdClusId:
        st = "" + str(d) + " " + str(dirichlet_model.docIdClusId[d]) + " \n"
        f.write(st)
    for d in dirichlet_model.deletedDocIdClusId:
        st = "" + str(d) + " " + str(dirichlet_model.deletedDocIdClusId[d]) + " \n"
        f.write(st)
    f.close()
