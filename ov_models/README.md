# Split and Combine models
```
split -b 90M model.bin part_
```

```
cat part_* > new_model.bin

```

compare the MD5
```
md5sum model.bin
md5sum new_model.bin
```