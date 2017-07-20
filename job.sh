##
#grid search for GRRU model
##

python nlp_task.py HyperLSTM -H 1000 -B 128 -T 100 -E 30 -Op Adam
python nlp_task.py HyperLSTM -H 1000 -B 128 -T 100 -iD True -E 30 -Op Adam


python nlp_task.py HyperLSTM -H 1000 -B 128 -T 100 -E 40 -Op Adam
python nlp_task.py HyperLSTM -H 1000 -B 128 -T 100 -iD True -E 40 -Op Adam

python nlp_task.py HyperLSTM -H 1000 -B 128 -T 100 -E 50 -Op Adam
python nlp_task.py HyperLSTM -H 1000 -B 128 -T 100 -iD True -E 50 -Op Adam

python nlp_task.py HyperLSTM -H 1000 -B 128 -T 100 -E 60 -Op Adam
python nlp_task.py HyperLSTM -H 1000 -B 128 -T 100 -iD True -E 60 -Op Adam

python nlp_task.py HyperLSTM -H 1000 -B 128 -T 100 -E 70 -Op Adam
python nlp_task.py HyperLSTM -H 1000 -B 128 -T 100 -iD True -E 70 -Op Adam

python nlp_task.py HyperLSTM -H 1000 -B 128 -T 100 -E 80 -Op Adam
python nlp_task.py HyperLSTM -H 1000 -B 128 -T 100 -iD True -E 80 -Op Adam

#python nlp_task.py HyperLSTM -H 1000 -B 128 -T 100 -iD True -Ta True -E 30 -Op Adam
#python nlp_task.py HyperLSTM -H 1000 -B 128 -T 100 -Ta True -E 30 -Op Adam




