import fasttext

class EntityResolutionFeatures():
    def __init__(self, ft_path:str='data/cc.en.50.bin'):
        self.ft_model = fasttext.load_model(ft_path)
    
    def features(self, comb_df:pd.DataFrame) -> pd.DataFrame:
        comb_df['jw_fn_distance'] = comb_df.apply()
