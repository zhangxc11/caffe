#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Last modified: 2015  1 13 21时18分32秒

"""docstring
"""

__revision__ = '0.1'

def main():
  names = ["C1"
          , "banner_pos"
          , "site_id"
          , "site_domain"
          , "site_category"
          , "app_id"
          , "app_domain"
          , "app_category"
          , "device_model"
          , "device_type"
          , "device_conn_type"
          , "C14"
          , "C15"
          , "C16"
          , "C17"
          , "C18"
          , "C19"
          , "C20"
          , "C21"]
  dims = [8, 8, 962, 856, 20, 873, 57, 21, 2696, 5, 5,
      834, 9, 10, 215, 5, 49, 147, 41]
  out = [50, 50, 200, 200, 50, 200, 80, 50, 400, 30, 30,
      200, 50, 20, 100, 30, 80, 150, 80]
  print '''name: "Net"
layers {
  name: "ctr"
  type: DATA
  top: "data"
  top: "label"
  data_param {
    source: "train_db"
    backend: LMDB
    batch_size: 100
  }
  include: { phase: TRAIN }
}
layers {
  name: "ctr"
  type: DATA
  top: "data"
  top: "label"
  data_param {
    source: "test_db"
    backend: LMDB
    batch_size: 100
  }
  include: { phase: TEST }
}
layers {
  name: "extend"
  type: EXTEND
  bottom: "data"'''
  for i in range(19):
    print '  top: \"' + names[i] + '\"'
  print '  extend_param {'
  for i in range(19):
    print '    dim:', dims[i]
    print '    pos:', i
  print '  }'
  print '}'
  for i in range(19):
    print '''layers {
  name: \"%s\"
  type: INNER_PRODUCT
  bottom: \"%s\"
  top: \"f%s\"
  blobs_lr: 1
  blobs_lr: 2
  inner_product_param {
    num_output: %d
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}''' % (names[i], names[i], names[i], out[i])
  print '''layers {
  name: "Concat"
  type: CONCAT
  top: "fea"'''
  for i in range(19):
    print '  bottom: \"f%s\"' % names[i]
  print '''  concat_param {
    concat_dim: 1
  }
}'''
  print '''layers {
  name: "relu"
  type: RELU
  bottom: "fea"
  top: "fea"
}
layers {
  name: "ip1"
  type: INNER_PRODUCT
  bottom: "fea"
  top: "ip1"
  blobs_lr: 1
  blobs_lr: 2
  inner_product_param {
    num_output: 2000
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layers {
  name: "relu1"
  type: RELU
  bottom: "ip1"
  top: "ip1"
}
layers {
  name: "ip2"
  type: INNER_PRODUCT
  bottom: "ip1"
  top: "ip2"
  blobs_lr: 1
  blobs_lr: 2
  inner_product_param {
    num_output: 5000
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layers {
  name: "relu2"
  type: RELU
  bottom: "ip2"
  top: "ip2"
}
layers {
  name: "drop2"
  type: DROPOUT
  bottom: "ip2"
  top: "ip2"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layers {
  name: "ip3"
  type: INNER_PRODUCT
  bottom: "ip2"
  top: "ip3"
  blobs_lr: 1
  blobs_lr: 2
  inner_product_param {
    num_output: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layers {
  name: "accuracy"
  type: ACCURACY
  bottom: "ip3"
  bottom: "label"
  top: "accuracy"
  include: { phase: TEST }
}
layers {
  name: "loss"
  type: SOFTMAX_LOSS
  bottom: "ip3"
  bottom: "label"
  top: "loss"
}'''


if __name__ == '__main__':
  main()
