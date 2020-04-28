from utils.data_cleaning import clean_for_content, clean_for_topic
from tqdm.autonotebook import tqdm

#alter these global variables for save paths
IN_DATA_PATH = '/home/sentiment/data_lake/twitter/processed/'
IN_DATA_SUFFIX = '.csv'
IN_DATA_SEP = ','
OUT_DATA_PATH = '/home/sentiment/data_lake/twitter/nlp_project_samples/'

def process_data(date):

    print("\nReading in {}".format(date))
    df = pd.concat([
        pd.read_csv(IN_DATA_PATH+'info_%s.csv' % date, lineterminator='\n'),
        pd.read_csv(IN_DATA_PATH+'text_%s.csv' % date, lineterminator='\n', usecols=['tweet_text'])
    ], axis=1)

    df = df[(df['country']=="US")&(df['lang']=="en")].reset_index(drop=True)

    print("Cleaning...")
    df['tweet_text_clean'] = [clean_for_content(elem) for elem in tqdm(df['tweet_text'].values)]

    print("Stemming...")
    df['stem_flag'] = df['reply_to_tweet_id'].isna() # Don't clean for topic reply tweets
    df['tweet_text_stemmed'] = [
        clean_for_topic(elem) if flag==True else '' for elem, flag in tqdm(zip(df['tweet_text_clean'], df['stem_flag']), total=len(df['stem_flag']))
    ]
    del df['stem_flag']

    df.to_csv(OUT_DATA_PATH+str(date)+'.tsv', sep='\t', index=False)

    del df

date = sys.argv[1]

if __name__ == '__main__':
    process_data(date)
