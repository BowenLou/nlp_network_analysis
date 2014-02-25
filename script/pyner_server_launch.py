import os
import subprocess

"""
    launch pyner server
"""

# can also use other 4 or 7 classes models
print 'using PyNER, the Python interface to the Stanford Named Entity Recognizer'
cmd = 'java -mx1500m -cp stanford-ner-2014-01-04/stanford-ner.jar edu.stanford.nlp.ie.NERServer -loadClassifier stanford-ner-2014-01-04/classifiers/english.all.3class.distsim.crf.ser.gz -port 8080 -outputFormat inlineXML'

os.system(cmd)


