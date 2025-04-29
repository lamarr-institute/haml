Syntax highlighting for the HAML files (.hml) in vscode. More details about the fileformat can be found [here](https://github.com/lamarr-institute/haml)

# Install the extension

The latest version of the extension is part of the repository. You can install it via

```
code --install-extension FILENAME.vsix
```

If this does not work for whatever reason, please consider building the extension yourself. Details see below.

# Debug the extension

You can easily debug changes to the extension. Just navigate to the folder and start code
```
cd PATH_TO_EXTENSION
code .
```
and then you can start a debug extension of vscode via the run/debug command (F5). You can use the `test.hml` to check if syntax highlighting works as intended. 


# Build the extension

You can rebuild the extension via
```
cd PATH_TO_EXTENSION
npm install -g @vscode/vsce
vsce package
```
which should produce a file similar to 
```
haml-syntax-highlighting-0.0.1.vsix
```
that you can then install as described above. 

