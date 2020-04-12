import Australian, Abalone, Balance, Diabetes, Glass, Heart, Landsat, Letter, Segment, Sonar, SVMguide1, Vehicle, Vote, Vowel, Waveform3, Winequality

datasets = [Australian, Abalone, Balance, Diabetes, Glass, Heart, Landsat, Segment, Sonar, SVMguide1, Vehicle, Vowel, Vote, Winequality, Waveform3]

if __name__ == '__main__':
    epochs = 5
    '''
    with open('TestResult.txt', 'a') as f:
        f.write('Tested in {} epochs. \n'.format(epochs))
    '''

    for _ in range(epochs):
        Winequality.test('replace')






