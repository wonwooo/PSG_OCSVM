import argparse
from Experiment import Australian, Abalone, Balance, Diabetes, Glass, Heart, Landsat, Letter, Segment, Sonar, \
    SVMguide1, Vehicle, Vote, Vowel, Waveform3, Winequality


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', type=int, help='number of epochs to test in same dataset')
    parser.add_argument('--generation-type', type=str, default='replace', help='pseudo outlier generation type')
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = get_args()
    epochs = args['num_epochs']
    generation_type = args['generation_type']

    # FIXME. Letter는 매우 오래 걸린다.
    # datasets = [Australian, Abalone, Balance, Diabetes, Glass, Heart, Landsat, Segment, Sonar, SVMguide1, Vehicle,
    #             Vowel, Vote, Winequality, Waveform3]
    #
    datasets = [SVMguide1]
    with open('TestResult.txt', 'w') as f:
        f.write('Tested in {} epochs. \n'.format(epochs))

    for _ in range(epochs):
        for dataset in datasets:
            dataset.main(generation_type)






