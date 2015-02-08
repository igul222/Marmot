# Sentiment analysis web app server
spawn = require('child_process').spawn
uuid  = require('uuid')
path  = require('path')
fs    = require('fs')

express     = require('express')
app         = express()
http        = require('http').Server(app)
io          = require('socket.io')(http);
multiparty  = require('multiparty')

models = {}

app.set 'port', process.env.PORT || 5000
app.use require('morgan')('dev')

app.get '/', (req, res) ->
  res.render 'index.jade'

app.post '/train', (req, res) ->
  form = new multiparty.Form()
  form.parse req, (err, fields, files) ->
    id = uuid.v4()
    models[id] = {
      trained: false, 
      trainingProgress: '', 
      subscribers: [],
      # Don't let this be anything other than 'rntn' or 'logreg' or you're open
      # to remote code execution vulnerabilities.
      type: (if fields.model[0] == 'rntn' then 'rntn' else 'logreg')
    }

    _process = spawn(
      'python',
      [
        models[id].type + '.py',
        'train',
        files.file[0].path, 
        path.join(__dirname, id + '.modeldump')
      ]
    )

    updateModel = (data) ->
      console.log 'incoming: '+data.toString()
      str = data.toString()
      models[id].trainingProgress += str
      for sub in models[id].subscribers
        sub(models[id])

    _process.stdout.setEncoding('utf8')
    _process.stderr.setEncoding('utf8')
    _process.stdout.on 'data', updateModel
    _process.stderr.on 'data', updateModel

    _process.on 'close', (code) ->
      if code == 0
        models[id].trained = true
        for sub in models[id].subscribers
          sub(models[id])

    res.render 'train.jade', {modelId: id}

app.get '/:modelId/test', (req, res) ->
  modelId = req.params['modelId']
  res.render 'test.jade', {modelId: modelId, model: models[modelId], result: null}

app.post '/:modelId/test', (req, res) ->
  form = new multiparty.Form()
  form.parse req, (err, fields, files) ->
    modelId = req.param('modelId')

    outputPath = path.join(__dirname, uuid.v4() + '.modeloutput')

    _process = spawn(
      'python',
      [
        models[modelId].type + '.py',
        'test',
        path.join(__dirname, modelId + '.modeldump')
        files.file[0].path,
        outputPath
      ]
    )

    render = (result) ->
      res.render 'test.jade', {
        modelId: modelId,
        model: models[modelId],
        result: result
      }

    _process.on 'close', (code) ->
      if code == 0
        fs.readFile outputPath, (err, data) ->
          if err
            render('An error occurred (no output file produced).')
          else
            render(data)
      else
        render('An error occurred (process ended with code '+code+').')

io.on 'connection', (socket) ->
  socket.on 'subscribe', (id) ->
    if models[id]
      push = (model) ->
        socket.emit('update', model)
      models[id].subscribers.push(push)
      push(models[id])

# Start the server
http.listen app.get('port'), '0.0.0.0', (err) ->
  if err
    console.log 'Fatal error: '+err
  else
    console.log 'Express server listening on 0.0.0.0:' + app.get('port')