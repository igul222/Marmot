import marmot

def load_file(path):
    with open(path) as f:
        return [marmot.datasets.Trees.parse(l.strip('\n')) 
                for l in f.readlines()]

print "Loading data..."

training_data = marmot.datasets.Trees(
    load_file('data/stanford_treebank/train.txt'), 
    minibatch_size=25
)

validation_data = marmot.datasets.Trees(
    load_file('data/stanford_treebank/dev.txt'), 
    wordmap=training_data.wordmap
)

print "Training..."

recursive = marmot.layers.Recursive(
    word_vec_length=30,
    wordmap=training_data.wordmap,
    tensor=True
)
softmax   = marmot.layers.Softmax(prev_layer=recursive, n=5)
l2reg     = marmot.layers.L2Reg(prev_layer=softmax, reg_weight = 1e-4)

# Define a learning strategy
learning_rule = marmot.sgd.Adadelta(decay = 0.90, epsilon = 1e-6)
strategy = marmot.sgd.SGD(learning_rule=learning_rule)

# Initialize and run the training loop
marmot.train_loop(
    l2reg, 
    strategy, 
    training_data, 
    validation_data, 
    patience_factor=2, 
    validation_frequency=3
)