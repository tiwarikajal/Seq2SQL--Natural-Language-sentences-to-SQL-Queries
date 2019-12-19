from __future__ import unicode_literals, print_function, division
from baseline import baseline
from seq2sql import train as train_target
from seq2sql import test as test_target

print("Running baseline")
baseline.run_baseline()

print("Training target model")
train_target.train_seq2sql()

print("Testing target model")
test_target.test_seq2sql()
