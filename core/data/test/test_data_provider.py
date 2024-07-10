


import unittest

import pandas as pd
from torch import Tensor
import torch
from core.data.data_provider import ChunkType, DataProvider


class TestDataProvider(unittest.TestCase):
    def test_get_data(self):

        df = pd.DataFrame(
            columns=['date', 'value', 'chunk', 'chunk_type', 'dummy'],
            data=[
                ['2021-01-01', 1, -1, ChunkType.TRAINING, 21], 
                ['2021-01-02', 2, -1, ChunkType.TRAINING, 20], 

                ['2021-01-03', 3, 0, 1, 19], 
                ['2021-01-04', 4, 0, 1, 18], 
                
                ['2021-01-05', 5, 1, ChunkType.TRAINING, 17], 
                ['2021-01-06', 6, 1, ChunkType.TRAINING, 16], 
                ['2021-01-07', 7, 1, ChunkType.TRAINING, 15], 
                
                ['2021-01-08', 8, 2, ChunkType.TRAINING, 14], 
                ['2021-01-09', 9, 2, ChunkType.TRAINING, 13], 
                
                ['2021-01-10', 10, -1, ChunkType.TRAINING, 12], 
                
                ['2021-01-11', 11, 3, 2, 11],
                
                ['2021-01-12', 12, 4, 1, 10],
                ['2021-01-13', 13, 4, 1, 9],
                ['2021-01-14', 14, 4, 1, 8],
                ['2021-01-15', 15, 4, 1, 7],
                
                ['2021-01-16', 16, 5, ChunkType.TRAINING, 6],
                
                ['2021-01-17', 17, -1, ChunkType.TRAINING, 5],
                
                ['2021-01-18', 18, 6, 2, 4],
                ['2021-01-19', 19, 6, 2, 3],
                ['2021-01-20', 20, 6, 2, 2],
                ['2021-01-21', 21, 6, 2, 1],])

        try:
            DataProvider(df, ['value', 'foo'], 'none')
            self.fail("Should raise exception")
        except ValueError:
            pass

        data_provider = DataProvider(df, ['value', 'dummy'], 'none')

        self.assertEqual(data_provider.get_chunk_cnt('tr'), 3)
        self.assertEqual(data_provider.get_chunk_cnt('val'), 2)
        self.assertEqual(data_provider.get_chunk_cnt('test'), 2)

        try:
            data_provider.get_iterator('unknown')
            self.fail("Should raise exception")
        except ValueError:
            pass
        try:
            data_provider.get_chunk_cnt('unknown')
            self.fail("Should raise exception")
        except ValueError:
            pass

        tr_it = data_provider.get_iterator('tr')
        val_it = data_provider.get_iterator('val')
        test_it = data_provider.get_iterator('test')

        tr_list = list(tr_it)
        val_list = list(val_it)
        test_list = list(test_it)

        self.assertEqual(len(tr_list), 3)
        self.assertEqual(len(val_list), 2)
        self.assertEqual(len(test_list), 2)


        self.assertTrue(torch.equal(tr_list[0], 
                                    Tensor([[5, 17], [6, 16], [7, 15]])))
        self.assertTrue(torch.equal(tr_list[1], 
                                    Tensor([[8, 14], [9, 13]])))
        self.assertTrue(torch.equal(tr_list[2], 
                                    Tensor([[16, 6]])))
        

        self.assertTrue(torch.equal(val_list[0], 
                                    Tensor([[3, 19], [4, 18]])))
        self.assertTrue(torch.equal(val_list[1], 
                                    Tensor([[12, 10], [13, 9], [14, 8], [15, 7], ])))
        

        self.assertTrue(torch.equal(test_list[0], 
                                    Tensor([[11, 11]])))
        self.assertTrue(torch.equal(test_list[1], 
                                    Tensor([[18, 4], [19, 3], [20, 2], [21, 1]])))
    def test_normalization(self):

        df = pd.DataFrame(
            columns=['date', 'value', 'chunk', 'chunk_type', 'dummy'],
            data=[
                ['2021-01-01', 1, -1, 0, 21], 
                ['2021-01-02', 2, -1, 0, 20], 
                ['2021-01-03', 3, 0, 1, 19], 
                ['2021-01-04', 4, 0, 1, 18], 
                ['2021-01-05', 5, 1, 0, 17], 
                ['2021-01-06', 6, 1, 0, 16], 
                ['2021-01-07', 7, 1, 0, 15], 
                ['2021-01-08', 8, 2, 0, 14], 
                ['2021-01-09', 9, 2, 0, 13], 
                ['2021-01-10', 10, -1, 0, 12], 
                ['2021-01-11', 11, 3, 2, 11],
                ['2021-01-12', 12, 4, 1, 10],
                ['2021-01-13', 13, 4, 1, 9],
                ['2021-01-14', 14, 4, 1, 8],
                ['2021-01-15', 15, 4, 1, 7],
                ['2021-01-16', 16, 5, 0, 6],
                ['2021-01-17', 17, -1, 0, 5],
                ['2021-01-18', 18, 6, 2, 4],
                ['2021-01-19', 19, 6, 2, 3],
                ['2021-01-20', 20, 6, 2, 2],
                ['2021-01-21', 21, 6, 2, 1],])

        data_provider = DataProvider(df, ['value', 'dummy'], 'zscore')
        
        df = data_provider._df


