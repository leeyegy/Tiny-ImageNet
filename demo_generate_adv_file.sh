for attack in DeepFool
do 
for epsilon in 0.06275
do 
python data_generator.py --attack_method $attack  --epsilon $epsilon 
done 
done 
