"""creates feature of sequence files
Input: seqfile {0: regionTag, 1: spTag, 2: sequence, 4: seqLength
Output: Labelled example feature matrix, list of labelled species,
        corresponding orthologs with their species name {speciesList, featurevectorList},
        labels

(c) Faizy Ahsan
email: zaifyahsan@gmail.com"""


import sys
import logging
import numpy as np
import collections
import json

from Bio.Seq import Seq
from Bio.Alphabet import generic_dna

class FeatureFromSeq:

    def __init__(self):
        self.allseq = dict()
        self.produceAll(['A', 'C', 'G', 'T'], '', 6)
        self.allseq = collections.OrderedDict(sorted(self.allseq.items()))
        return

    """Reads featureFile line by line
    creates the four output variables
    1. labelled feature matrix
    2. list of labelled species
    3. OrthologDict{ labelled_feature_idx:
                                {species: species_list,
                                 X: orthologs_feature_matrix
                                 }
                            }
    4. labels"""
    def processFile(self, labelled_species_list, feature_filename, is_it_testfile):

        labelled_features = dict()
        labelled_species = dict()
        all_orthologs_info = dict()
        labels = dict()

        fid = open( feature_filename , 'r')
        for line in fid:
            logging.debug('line: %s', line)
            line = line.strip().split()
            region_tag = line[0]
            # TODO: Temporary fix for new file format (4.2k). Uncomment if using old seqadded files
            #labelled_example_idx = int(region_tag.split('_')[1].split('region')[1])
            labelled_example_idx = int(region_tag.split('_')[3].split('region')[1].split("-")[0])
            logging.debug('example_id: %s', labelled_example_idx)
            sptag = line[1].split('.')[0].replace('_', '')
            seq = line[2]
            feature_vector = self.kmerCount(seq)
            
            region_label = 1.0 if 'pos' in region_tag else 0.0

            # check if region belongs to labelled species
            if sptag in labelled_species_list or is_it_testfile:
                labels[labelled_example_idx] = region_label
                labelled_features[labelled_example_idx] = feature_vector
                labelled_species[labelled_example_idx] = sptag

            else:
                if labelled_example_idx not in all_orthologs_info.keys():
                    all_orthologs_info[labelled_example_idx] = {'species': np.asarray(sptag, dtype=np.str),
                                                                'X': feature_vector}

                else:
                    all_orthologs_info[labelled_example_idx]['species'] = np.append(
                                                                    all_orthologs_info[labelled_example_idx]['species'],
                                                                    sptag )
                    all_orthologs_info[labelled_example_idx]['X'] = np.vstack(
                                                                    (all_orthologs_info[labelled_example_idx]['X'],
                                                                    feature_vector)
                                                                    )

        if is_it_testfile:
            feature_info = {'labelled_examples': labelled_features,
                        'labels': labels}
        else:
            feature_info = {'labelled_examples': labelled_features,
                            'labelled_species': labelled_species,
                            'ortho_info': all_orthologs_info,
                            'labels': labels}



        return feature_info

    def produceAll(self, S, prefix, k):
        # allseq = self.allseq
        if k == 0:
            self.allseq[prefix] = 0
            return ['']

        for i in range(0, len(S)):
            newprefix = prefix
            newprefix = newprefix + S[i]
            self.produceAll(S, newprefix, k - 1)

    def kmerCount(self, seq):
        for index in range(0, len(seq)-5):
            word = seq[index: index+6]
            self.allseq[word] += 1
            dna = Seq(word, generic_dna)
            rev_word = [ bp for bp in dna.reverse_complement() ]
            rev_word = ''.join(rev_word)
            self.allseq[ rev_word ] += 1

        feature_vector = [ v for k,v in self.allseq.items() ]
        feature_vector = np.asarray(feature_vector, dtype=np.float)

        self.allseq = { k: 0 for k in self.allseq }

        return feature_vector


if __name__ == '__main__':

    # logging.basicConfig( level=logging.DEBUG,
    # 			format="%(asctime)s.%(msecs)d %(levelname)s %(module)s: %(message)s")


    if len(sys.argv) < 3:
        print ('python file.jsony seqfile is_it_testfile{ 0 or 1}')
        exit(1)

    filename = sys.argv[1]
    is_it_testfile = bool(int(sys.argv[2]))

    # if is_it_testfile:
    #     print 'yes'
    # else:
    #     print 'no'
    #
    # exit(1)

    featureObj = FeatureFromSeq(filename, is_it_testfile)

    # [ print(v) for v in featureObj.kmerCount('AAAAAT') ]
    # print ( featureObj.kmerCount('TTTTTTTT'))

    feature_info = featureObj.processFile(['hg38'])

    print ('feature_info size: %d' %(sys.getsizeof(feature_info) ) )

    exit(1)

    # json.dump( feature_info, open(filename+'.json', 'wb'))

    with open(filename+'.json', 'w') as fp:
        json.dump( feature_info, fp)

    # store all the four dictionaries
    # with open(filename+'-labelled_examples.json', 'w') as fp:
    #     json.dump( feature_info['labelled_examples'], fp)
    # with open()
    # json.dumps( feature_info['labels'], open(filename+'-labels.json', 'w'))
    #
    # if not is_it_testfile:
    #     json.dump(feature_info['labelled_species'], open(filename + '-labelled_species.json', 'w'))
    #     json.dump(feature_info['ortho_info'], open(filename + '-ortho_info.json', 'w'))

    #  store all the four dictionaries
    # json.dump( feature_info['labelled_examples'], open(filename+'-labelled_examples.json', 'wb'))
    # json.dump( feature_info['labels'], open(filename+'-labels.json', 'wb'))
    #
    # if not is_it_testfile:
    #     json.dump(feature_info['labelled_species'], open(filename + '-labelled_species.json', 'wb'))
    #     json.dump(feature_info['ortho_info'], open(filename + '-ortho_info.json', 'wb'))

    # print (feature_info )
    #
    # for k,v in feature_info['ortho_info'].items():
    #     print (k, len(v['X']))
