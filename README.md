# tdt4215-project

To set up the program, please do the following: <br>

1. Clone the repository
2. Install dependencies using <code>pip install -r requirements.txt</code> from the <code>tdt4215-project</code> (root) folder. It is recommended, but not required, to use a virtual python environment for this.

To evaluate, run <code>python evaluate.py</code>, then input one of the available methods in the command prompt (i.e. <code>bas</code> for baseline). Other optional actions are:

- Change parameter inputs of various methods when evaluating by altering them in <code>evaluate.py</code>'s <code>main</code> method.
- Toggle verbose logging by commenting out <code>os.environ["DEBUG"] = "1"</code> in <code>evaluate.py</code>.
- Toggle user sampling by altering <code>sample_users</code> in <code>evaluate.py</code>.
- Include additional stop words by appending them to the <code>danish_names</code> list in <code>content.py</code>.

## Dataset

The dataset used for the project is EB-NeRD small. More information about the dataset can be found on [Ekstra Bladet's documentation pages](https://recsys.eb.dk/dataset/).

## Recommendation methods used

### Baseline

The baseline method uses a simple popularity function. It combines pageviews (times an article has been clicked) with the average read time per article, in an attempt to only capture the most gripping articles. Using the baseline method is as simple as calling the <code>compute_recommendations_for_users</code> function in <code>baseline.py</code>, which returns a dataframe with baseline recommendations.

### Collaborative filtering

### Content-based filtering

For our content-based filtering, we use a Bag of Words (BoW) approach. That is, the words that make up each article (title, body, category, etc.) are stored in their own column. Then, the words in each article are counted and stop words removed. Depending on whether the method is run with the <code>use_lda</code> flag or not, either a TF-IDF or LDA matrix will be used to represent all the articles. Linear kernel is then used to compute the similarity between the user feature vector and article vectors in the matrix. To evaluate either approach, input <code>tfi</code> or <code>lda</code> when running <code>evaluate.py</code>, respectively.

The <code>content.py</code> file is used for both approaches. To run the method in isolation, use the <code>compute_recommendations_for_users</code> function. Example usage of this can be seen in <code>content.py</code> or <code>evaluate.py</code>'s <code>main</code> methods. Both approaches come with a range of parameters that can be adjusted, most of which with set default values. Importantly, <code>n_topics</code> only affects the result if <code>use_lda</code> is set to <code>True</code>.

### Hybrid

## Evaluation

### Accuracy measurement

### Beyond accuracy metric

To evaluate the recommendation methods beyond accuracy, we utilize CodeCarbon's <code>EmissionsTracker</code> library. This library offers a simple way to estimate carbon emissions based on how much electricity is used by your device while running the methods. For our usage, we wrap the part of the <code>main</code> function in <code>evaluate.py</code> that computes recommendations with an <code>EmissionTracker</code>. This tracker logs emissions as the program runs, and displays the final carbon count after recommendations are generated. You can read more about CodeCarbon on [their website](https://codecarbon.io/).
