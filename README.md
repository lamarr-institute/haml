# Hyper-YAML (HAML)

<p align="center">
  <img src="logo.png" alt="haml logo" width="200"/>
</p>

HAML is an extension of YAML providing syntax to make parts of the file optional
or generate values.
This is particularly useful for generating YAML files defining the hyperparameters
of ML experiments.

## Installation



## Syntax

### Choice List

Inline choice lists:
```
model:
  loss: cross-entropy
  dropout_p: {{ 0.0 || 0.05 }}
  norm: channel 
  activation: elu
  mlp_size: {{ 128 || 512 }}
  name: rutime
```

Multi-line choice lists:
```
channels: {{ ["E1-M2", "E2-M1"]
|| ["E1-M2", "E2-M1", "1-F", "1-2", "2-F"]
|| ["E1-M2", "E2-M1", "1-F", "1-2", "2-F", "Resp Rate", "Pulse Waveform", "Heart Rate"]
}}
```

Weighted choice lists:
```
channels: {{2% ["E1-M2", "E2-M1"]
||3% ["E1-M2", "E2-M1", "1-F", "1-2", "2-F"]
||5% ["E1-M2", "E2-M1", "1-F", "1-2", "2-F", "Resp Rate", "Pulse Waveform", "Heart Rate"]}}
```

Optional blocks:
```
{{
option: debug
||}}
```

Multiple choice lists:
```
{{1-2%
  key: foo
  value: 3
||
  key: bar
  value: 4
||
  key: baz
  value: -29
}}
```

Random Values:
```
intensity: {{%normal(loc=10, scale=2)%}}
saturation: {{%uniform(low=0, high=10)%}}
num-scans: {{%integers(high=20)%}}
```
