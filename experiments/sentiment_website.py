import math
import numpy
import theano
import marmot
import random

from flask import Flask, request, Response, redirect, url_for
app = Flask(__name__)

def make_page(html):
    return '''
    <!doctype html>
    <html>
        <head>
            <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/3.3.2/css/bootstrap.min.css" />
            <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/3.3.2/css/bootstrap-theme.min.css" />
        </head>
        <body class="container">
    ''' + html + '''
        </body>
    </html>
    '''


@app.route('/')
def welcome():
    return make_page('''
        <div class="row">
            <div class="col-sm-6">
                <h1>Logistic regression</h1>
                <hr/>
                <form action="/logistic" method="post" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="file">Upload a tsv file with columns <code>label</code>, <code>string</code> (tokens separated by spaces):</label>
                        <input type="file" name="file" class="form-control">
                    </div>
                    <input type="submit" value="Train" class="btn btn-primary">
                </form>
            </div>

            <div class="col-sm-6">
                <h1>RNTN</h1>
                <hr/>
                <form action="/rntn" method="post" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="file">Upload a file with labelled parse trees:</label>
                        <input type="file" name="file" class="form-control">
                    </div>
                    <input type="submit" value="Train" class="btn btn-primary">
                </form>
            </div>
        </div>
        <hr/>
        be patient; both models take several minutes to train.
    ''')

def to_wordbags(strings, wordmap):
    def bagify(string):
        bag = numpy.zeros((len(wordmap),), dtype=theano.config.floatX)
        words = string.split(' ')
        for word in words:
            if word in wordmap:
                bag[wordmap[word]] += 1.0
            else:
                bag[wordmap['UNK']] += 1.0
        return bag / len(words)
    return map(bagify, strings)

logistic_model = {}
@app.route('/logistic', methods=['POST'])
def logistic():
    global logistic_model

    data = [l.strip('\n').split("\t") for l in request.files['file'].readlines()]
    random.shuffle(data)

    labels = [d[0] for d in data]
    strings = [d[1] for d in data]

    # Split into training/validation sets
    split = int(math.floor(len(data) * 0.9))

    labelmap = {}
    labelmap_len = 0
    for label in labels[:split]:
        if label not in labelmap:
            labelmap[label] = labelmap_len
            labelmap_len += 1

    wordmap = {'UNK': 0}
    wordmap_len = 0
    for string in strings[:split]:
        words = string.split(' ')
        for word in words:
            if word not in wordmap:
                wordmap[word] = wordmap_len
                wordmap_len += 1

    labels = [labelmap[l] for l in labels]
    word_bags = to_wordbags(strings, wordmap)

    training_data = marmot.datasets.Simple(word_bags[:split], labels[:split], minibatch_size=128)
    validation_data = marmot.datasets.Simple(word_bags[split:], labels[split:])

    inputs  = marmot.layers.Input(len(wordmap))
    softmax = marmot.layers.Softmax(prev_layer=inputs, n=len(labelmap))

    learning_rule = marmot.sgd.Adadelta(decay = 0.90, epsilon = 1e-4)
    strategy = marmot.sgd.SGD(learning_rule=learning_rule)

    marmot.train_loop(
        softmax,
        strategy,
        training_data,
        validation_data,
        min_patience=100,
        patience_factor=1.1,
        validation_frequency=3
    )

    logistic_model['model'] = softmax
    logistic_model['wordmap'] = wordmap
    logistic_model['labelmap'] = labelmap

    return redirect(url_for('test_logistic'))

@app.route('/test_logistic', methods=['GET', 'POST'])
def test_logistic():
    if request.method == 'GET':
        return make_page('''
            <h1>Test logistic regression model</h1>
            <hr/>
            <form action="/test_logistic" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="sentence">Enter a sentence:</label>
                    <input type="text" name="sentence" class="form-control" />
                </div>
                <input type="submit" value="Test" class="btn btn-primary">
            </form>
        ''')
    else:
        wordbags = to_wordbags([request.form['sentence']], logistic_model['wordmap'])

        test_data = marmot.datasets.Simple(wordbags, [7], minibatch_size=1)

        result_idx = logistic_model['model']._y_pred(test_data).eval()[0][0]
        result = None
        for word, idx in logistic_model['labelmap'].iteritems():
            if idx == result_idx:
                result = word

        return make_page('''
            <h1>Test logistic regression model</h1>
            <hr/>
            <div class="alert alert-success" role="alert">Classified as <strong>''' + result + '''</strong>.</div>
            <form action="/test_logistic" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="sentence">Enter a sentence:</label>
                    <input type="text" name="sentence" class="form-control" />
                </div>
                <input type="submit" value="Test" class="btn btn-primary">
            </form>
        ''')

rntn_model = {}
@app.route('/rntn', methods=['POST'])
def rntn():
    global rntn_model

    trees = [marmot.datasets.Trees.parse(l.strip('\n')) for l in request.files['file'].readlines()]

    split = int(math.floor(len(trees) * 0.9))
    training_data = marmot.datasets.Trees(trees[:split], minibatch_size=25)
    validation_data = marmot.datasets.Trees(trees[split:], wordmap=training_data.wordmap, minibatch_size=25)

    recursive = marmot.layers.Recursive(
        word_vec_length=30,
        wordmap=training_data.wordmap
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
        patience_factor=1.1, 
        validation_frequency=3
    )

    rntn_model['model'] = softmax
    rntn_model['wordmap'] = training_data.wordmap

    return redirect(url_for('test_rntn'))

@app.route('/test_rntn', methods=['GET', 'POST'])
def test_rntn():
    if request.method == 'GET':
        return make_page('''
            <h1>Test RNTN model</h1>
            <hr/>
            <form action="/test_rntn" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="sentence">Enter a labelled parse tree (labels can be incorrect; they aren't used for anything but the parser expects them):</label>
                    <input type="text" name="sentence" class="form-control" />
                </div>
                <input type="submit" value="Test" class="btn btn-primary">
            </form>
        ''')
    else:
        tree = marmot.datasets.Trees.parse(request.form['sentence'])
        dataset = marmot.datasets.Trees([tree], wordmap=rntn_model['wordmap'])

        output = rntn_model['model']._y_pred(dataset).eval()

        nodes = tree.traverse(str)
        results = "\n".join([nodes[i] + ": " + str(output[i][0]) for i in xrange(len(nodes))])

        return make_page('''
            <h1>Test RNTN model</h1>
            <hr/>
            <form action="/test_rntn" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="sentence">Enter a labelled parse tree (labels can be incorrect; they aren't used for anything but the parser expects them):</label>
                    <input type="text" name="sentence" class="form-control" />
                </div>
                <input type="submit" value="Test" class="btn btn-primary">
            </form>
            Results:
            <pre>''' + results + '''</pre>
        ''')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')