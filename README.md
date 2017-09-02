# env-context-detection
Code for "Environmental context classification using GNSS measurements from a smartphone" report as part of MSc project


Running code to check results

Running voting.py will output the voting method results. To choose which training and test sets are used - uncomment the marked relevant sections at the top of the file. The default is training and testing on Greenwich data.

Running stack_model.py will output results for stacking method as well as the underlying base methods. To choose which training and test sets are used - uncomment the marked relevant sections at the top of the file. To choose the meta classifier - uncomment the desired one in the section marked. The default is training and testing on Greenwich data using radial neighbours meta classifier.
