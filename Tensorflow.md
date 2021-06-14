# Basic Operations
Concatenation
```python
result = tf.concat(tensors, 1) # 1 is the dimension to concatenate
```

# Check saved checkpoint variables
```python
>>> import tensorflow as tf
>>> init_checkpoint = './bert_model.ckpt'
>>> init_vars = tf.train.list_variables(init_checkpoint)
>>> sess = tf.Session()
>>> print(init_vars)
```

# Freeze graph
```bash
python ~/github/tensorflow/tensorflow/python/tools/freeze_graph.py \
    --input_meta_graph=./tf_models/imdb_lstm.ckpt.meta \
    --input_checkpoint=./tf_models/imdb_lstm.ckpt \
    --output_graph=./tf_models/imdb_lstm_tf_frozen.pb \
    --output_node_names="model_1/logits" \
    --input_binary=true
```
Get model info
```python
def get_keras_model_info():
    model = keras.models.load_model('imdb_lstm_ep_5.h5')
    node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    print("\n".join(node_names))

def get_tf_model_info():
    sess = tf.Session()
    saver = tf.train.import_meta_graph('tf_models/imdb_lstm.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('tf_models/'))
    node_names = [n.name for n in sess.graph.as_graph_def().node]
    print("\n".join(node_names))
```

# Show graph in Jupyter
Refer to https://stackoverflow.com/questions/38189119/simple-way-to-visualize-a-tensorflow-graph-in-jupyter
```python
from IPython.display import clear_output, Image, display, HTML
import numpy as np

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
```