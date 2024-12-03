import Contant as con
import math
import torch.nn.functional as F

class Model:
    def __init__(self, args):
        self.alpha = args.alpha
        self.beta = args.beta
        self.delta = args.delta
        self.weight = args.weight
        self.time_elapsed = args.time_elapsed
        self.epsilon = args.epsilon
        self.applyICF = args.applyICF
        self.applyBERT = args.applyBERT

        self.word_wid_map = {}  # word, Assigned ID :    updated by Document
        self.wid_word_map = {}
        self.wid_docId = {}  # wordID, documentId:    updated by Document
        self.active_clusters = {}  # clusterID -> [ cn, cw, cwf, cww, cd, csw]
        self.active_documents = {}  # {documentId, Document}
        self.word_counter = {0: 0}
        self.widClusid = {}  # {wordID ,clusterID }: to know that how many cluster this word has occured
        self.docIdClusId = {}  # {documentID , clusterID} Cluster assignments of each document
        self.deletedDocIdClusId = {}  # those documents which are deleted while deleting the cluster, #this DS will be utilized to print output
        self.docEmbedding = {}
        self.docEmbedding = {0: 0}
        self.cluster_counter = {0: 0}
        self.currentTimestamp = 0
        self.cluster_num = {}

    def updatecurrentTimestamp(self):
        self.currentTimestamp += 1

    def sampleCluster(self, document):
        self.updatecurrentTimestamp()
        self.check_cluster_to_merge()
        clusIdOfMaxProb = -1
        clusMaxProb = 0.0

        N = self.active_documents.__len__() + 1  # number of maintained documents, some documents might be deleted from cluster
        VintoBETA = float(self.beta) * float(self.wid_docId.__len__())
        beta_sum = 0.0
        count_related_clusters = 0

        pro_sum = 0
        clusProb_list = {}
        for clusId in self.active_clusters:
            CF = self.active_clusters[clusId]
            cluster_wids = CF[con.I_cwf].keys()
            doc_wids = document.widFreq.keys()
            common_wids = self.intersection(cluster_wids, doc_wids)

            v_size = float(CF[con.I_cwf].__len__())
            v_size = v_size + (doc_wids.__len__() - common_wids.__len__())
            VintoBETA = float(self.beta) * v_size
            beta_sum += VintoBETA
            count_related_clusters += 1

            numOfDocInClus = CF[con.I_cn].__len__()
            eqPart1 = float(numOfDocInClus) / float((N - 1 + self.alpha * N))
            eqPart2 = 1.0

            numOfWordsInClus = CF[con.I_csw]
            i = 0  # represent word count in document
            for w in document.widFreq:
                widFreqInClus = 0
                if w in CF[con.I_cwf]:  # if the word of the document exists in cluster
                    widFreqInClus = CF[con.I_cwf][w]

                icf = 1.0
                if self.applyICF == True:  # This condition is used to control parameters by main method
                    icf = self.ICF(w)

                freq = document.widFreq[w]
                for j in range(freq):
                    eqPart2Nominator = (widFreqInClus * icf + self.beta + j)
                    eqPart2Denominator = (numOfWordsInClus + VintoBETA + i)  # 应该是加法
                    eqPart2 *= (eqPart2Nominator / eqPart2Denominator)
                    i += 1
                    eqPart2 *= i
                    eqPart2 *= 1 / (j + 1)

            clusProb = eqPart1 * eqPart2

            clusProb_list[clusId] = clusProb
            pro_sum += clusProb
            # end for , all probablities of existing clusters have been calculated
        if count_related_clusters > 0:
            VintoBETA = float(beta_sum) / float(count_related_clusters)
            # need to calculate probablity of creating a new cluster
        eqPart1 = (self.alpha * N) / (N - 1 + self.alpha * N)
        eqPart2 = 1.0
        i = 0  # represent word count in document
        for w in document.widFreq:
            freq = document.widFreq[w]

            for j in range(freq):
                eqPart2Nominator = (self.beta + j)
                eqPart2Denominator = (VintoBETA + i)
                eqPart2 *= (eqPart2Nominator / eqPart2Denominator)
                i += 1
                eqPart2 *= i
                eqPart2 *= 1 / (j + 1)

        probNewCluster = eqPart1 * eqPart2
        pro_sum += probNewCluster
        if (self.applyBERT == True):  # to control applying CWW from main method
            for clusId in self.active_clusters:
                CF = self.active_clusters[clusId]
                clusProb = (self.weight) * (
                        clusProb_list[clusId] / pro_sum) + (1 - self.weight) * self.cos_similar(document.embedding, CF[con.I_cce])

                if clusProb > clusMaxProb:
                    clusMaxProb = clusProb
                    clusIdOfMaxProb = clusId
        probNewCluster = (self.weight) * (probNewCluster / pro_sum) + (1 - self.weight) * self.delta
        if clusMaxProb >= probNewCluster:
            self.addDocumentIntoClusterFeature(document, clusIdOfMaxProb)
        else:
            self.createNewCluster(document)

    def check_cluster_to_merge(self):
        active_clusters_copy = self.active_clusters.copy()
        clusters_to_delete = []

        for clusId in active_clusters_copy:
            CF = active_clusters_copy[clusId]
            if (self.currentTimestamp - CF[con.I_cl]) > self.time_elapsed:
                temp_Max_Sim = -1
                temp_index = clusId
                for j in active_clusters_copy:
                    if clusId == j:
                        continue
                    temp_cluster = active_clusters_copy[j]
                    dis_temp = self.cos_similar(CF[con.I_cce], temp_cluster[con.I_cce])
                    if dis_temp > temp_Max_Sim:
                        temp_Max_Sim = dis_temp
                        temp_index = j
                if temp_Max_Sim > self.epsilon:
                    self.merger_clusters(clusId, temp_index)
                clusters_to_delete.append(clusId)

        for clusterID in clusters_to_delete:
            self.deleteOldCluster(clusterID, active_clusters_copy[clusterID])





