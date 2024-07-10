import unittest
from pandas import DataFrame
from core.data.technical_indicators.indicators import IndicatorParameterDescription, IndicatorPrototype


class DummyIndicator(IndicatorPrototype):

    def __init__(self, skip_f):
        super().__init__("dummy", [
                IndicatorParameterDescription("a", 0, 10, 4, 'int'),
                IndicatorParameterDescription("b", -1, 1, 0.2, 'float'),
            ], skip_f, norm_factor=3)
        self.params = self._params

    def calculate(self, params: dict, df: DataFrame):
        return params['b'] + df['w'].rolling(window=params['a']).sum()



class TestIndicators(unittest.TestCase):
    def test_hash(self):
        ind0 = DummyIndicator(None).get_descriptor().create_indicator({'a': 5, 'b': 0})
        ind1 = DummyIndicator(None).get_descriptor().create_indicator({'a': 5, 'b': 1})
        ind2 = DummyIndicator(None).get_descriptor().create_indicator({'a': 5, 'b': 0})
        
        self.assertNotEqual(hash(ind0), hash(ind1))
        self.assertEqual(hash(ind0), hash(ind2))
        


    def test(self):
        indicator_prototype = DummyIndicator(skip_f=None)

        desc = indicator_prototype.get_descriptor()
        self.assertListEqual(indicator_prototype.params,
                             desc.get_parameter_descriptions())

        params = dict([(p.name, p.default_val) for p in
                       desc.get_parameter_descriptions()])
        self.assertEqual(4, params['a'])
        self.assertEqual(0.2, params['b'])

        indicator = desc.create_indicator(params)
        self.assertEqual(0, indicator.get_skip_cnt())
        self.assertEqual(3, indicator.get_norm_factor())

        df = DataFrame(columns=['w'])
        df['w'] = range(15)
        indicator.apply_to_df(df, hash(indicator), 0, 15)
        indicator_prototype = DummyIndicator(skip_f='a')
        desc = indicator_prototype.get_descriptor()
        self.assertListEqual(indicator_prototype.params,
                             desc.get_parameter_descriptions())

        indicator = desc.create_indicator({'a': 5, 'b': 0})
        self.assertEqual(5, indicator.get_skip_cnt())
        self.assertEqual(3, indicator.get_norm_factor())

        df = DataFrame(columns=['w'])
        df['w'] = range(15)
        indicator.apply_to_df(df, hash(indicator), 0, 15)
