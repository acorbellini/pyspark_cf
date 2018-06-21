from mapsplotlib import mapsplot as mplt
import pandas as pd
if __name__ == '__main__':
    mplt.register_api_key('AIzaSyBuhS-obrP54G_ToqQAn3rnDY4hLxjy3Z4')
    with open("../dataset/yelp_academic_dataset_business.json") as business_file:
        df = pd.read_json(business_file, lines=True)
    df = df[df['city'] == 'Phoenix']
    mplt.density_plot(df['latitude'], df['longitude'])