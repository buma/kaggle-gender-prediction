language: PYTHON
name: "spear"

variable {
 name: "ntrees"
 type: INT
 size: 1
 min: 20
 max: 60
}

variable {
 name: "max_f"
 type: INT
 size: 1
 min: 1
 max: 362
}

variable {
 name: "criterion"
 type: ENUM
 size: 1
 options: "gini"
 options: "entropy"
}
