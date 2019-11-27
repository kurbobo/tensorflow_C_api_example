# Example TensorFlow C API

I made the same as on link 
https://github.com/Neargye/hello_tf_c_api

I needed to run a .pb file with tensorflow model for this reason I dowloaded my pb file to /models/saved_model.pb, 
added it to CMakeLists.txt 

```
configure_file(models/saved_model.pb ${CMAKE_CURRENT_BINARY_DIR}/saved_model.pb COPYONLY)
```

After that I added inputs, which have to be added to layers with curtain name

Names of layers and dimensions of expected inputs and outputs u can find using 

```
saved_model_cli show --dir=/path/to/model/dir --all
```

In my example I have 3 inputs and 2 outputs
