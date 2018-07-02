# word_embeddings_hp
Training and exploring Word2Vec and FastText on Harry Potter books corpus
<img src="https://vignette.wikia.nocookie.net/harrypotter/images/f/fd/Hallows.svg/revision/latest?cb=20100212055050" width=100>

## Data
http://www.glozman.com/textpages.html

## Libraries
<a href="https://radimrehurek.com/gensim/"> gensim </a>
<a href="https://www.nltk.org/"> NLTK </a>
<a href="http://scikit-learn.org/"> scikit-learn </a>

## Demo
### semantic similarities
```python
fasttext_sg.most_similar("wizard")  # FastText with Skip-Gram
>>> [('lizard', 0.9143307209014893),
 ('wizardkind', 0.9103794097900391),
 ('wiz', 0.9028722047805786),
 ('wizardry', 0.8854329586029053),
 ('triwizard', 0.8668864369392395),
 ('dark_wizard', 0.8515946865081787),
 ('wizard_chess', 0.8342143893241882),
 ('greatest_wizard', 0.8299335241317749),
 ('balding_wizard', 0.8259945511817932),
 ('wizard_prison', 0.8133100271224976)]
 
 w2v_sg.most_similar("wizard")  # Word2Vec with Skip-Gram
 >>> [('witch', 0.7799991369247437),
 ('man', 0.7468042969703674),
 ('house_elf', 0.7278867959976196),
 ('witch_or', 0.7212192416191101),
 ('boy', 0.6939235925674438),
 ('lady', 0.6722564101219177),
 ('skinny', 0.6555918455123901),
 ('child', 0.652462363243103),
 ('thief', 0.6514541506767273),
 ('muggle_born', 0.6514096260070801)]
```
### Word2Vec (Skip-Gram): 2-D plot of semantic similarities
<p> t-SNE used for dimensionality reduction </p>
<img src="https://preview.ibb.co/d4hiKy/download.png">

## Useful Tutorials
### Siraj Raval Word2Vec Live tutorial
<p> https://youtu.be/pY9EwZ02sXU </p>
<p> https://github.com/llSourcell/word_vectors_game_of_thrones-LIVE </p>
### Chris McCormick Skip-Gram for theory
<p>http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/</p>
