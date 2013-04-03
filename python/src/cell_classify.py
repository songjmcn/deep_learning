import numpy as np
import dataset.create_dataset as dataset
import theano
import theano.tensor as T
import learning.logistic as ln
import time

def trim(sstr):
    ystr=sstr.lstrip()
    ystr=ystr.rstrip()
    ystr=ystr.strip()
    ystr=ystr.strip('\n')
    return ystr
def label_to_int(label):
    if label=='centromere':
        return 1
    elif label=='coarse_speckled':
        return 2
    elif label=='cytoplasmatic':
        return 3
    elif label=='fine_speckled'   :
        return 4
    elif label=='homogeneous':
        return 5
    elif label=='nucleolar':
        return 6

def share_data(data_set,borrow=True):
    
    input=theano.shared(np.asarray(data_set[0],dtype=theano.config.floatX),borrow=borrow)
    output=theano.shared(np.asarray(data_set[1],dtype=theano.config.floatX),borrow=borrow)
    return input,T.cast(output,'int32')
def np_dataset(dataset):
    input=np.array(dataset[0], dtype='float32')
    output=np.array(dataset[1],dtype='int64')
    r=tuple([input,output])
    return r
def cell_train_and_test(dataset_path,learning_rate=0.13, n_epochs=1000):
    theano.config.blas.ldflags="-lopenblas"
    train_dataset,test_dataset=dataset.load_dateset(dataset_path)
    train_dataset=np_dataset(train_dataset)
    print(train_dataset[0].dtype)
    print(train_dataset[1].dtype)
    test_dataset=np_dataset(test_dataset)
    train_input,train_output=share_data(train_dataset)
    test_input,test_output=share_data(test_dataset)
    batch_size=20
    n_train_batches=train_input.get_value(borrow=True).shape[0]/batch_size-1
    n_test_batches=test_input.get_value(borrow=True).shape[0]/batch_size-1
    print '... building the model'
    index=T.lscalar()
    x=T.matrix('X')
    y=T.ivector('Y')
    classifier = ln.LogisticRegression(input=x, n_in=80 * 80, n_out=6)
    cost = classifier.negative_log_likelihood(y)
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_input[index * batch_size: (index + 1) * batch_size],
                y: test_output[index * batch_size: (index + 1) * batch_size]})
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]
    train_model = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_input[index * batch_size:(index + 1) * batch_size],
                y: train_output[index * batch_size:(index + 1) * batch_size]})
    print '... training the model'
    # early-stopping parameters
    patience = 721  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    best_params = None
    best_validation_loss = np.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = epoch * n_train_batches + minibatch_index
            test_losses = [test_model(i)
                for i in xrange(n_test_batches)]
            test_score = np.mean(test_losses)
            print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
if __name__=='__main__':
    dataset_path='d:/data/cell_dataset.gz'
    cell_train_and_test(dataset_path)