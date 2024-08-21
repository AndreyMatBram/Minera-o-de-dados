# -----------------------------------------------------------------------------
# (C) 2024 Andre Conde (andre.conde100@gmail.com)  (MIT License)
# -----------------------------------------------------------------------------


from functools import lru_cache

import numpy as np


class BaseMapper:
    """
    Base class for mappers.
    """

    __column__ = None
    __data__ = None

    def __init__(self):
        pass

    @classmethod
    def map(cls, value):
        if cls.__data__ is None:
            raise NotImplementedError("The __data__ attribute must be implemented.")

        if value == 'unknown':
            return np.nan
        elif value not in cls.__data__.keys():
            raise ValueError(f"Value {value} not found in {cls.__column__} mapper.")

        return cls.__data__[value]

    @classmethod
    def revert(cls, value):
        if cls.__data__ is None:
            raise NotImplementedError("The __data__ attribute must be implemented.")

        if np.isnan(value):
            return 'unknown'
        elif value not in cls.__data__.values():
            raise ValueError(f"Value {value} not found in {cls.__column__} mapper.")

        return list(cls.__data__.keys())[list(cls.__data__.values()).index(value)]

    @classmethod
    def map_list(cls, values):
        return [cls.map(value) for value in values]

    @classmethod
    def revert_list(cls, values):
        return [cls.revert(value) for value in values]

    @staticmethod
    @lru_cache
    def get_mapper(name: str):
        classes = BaseMapper.__subclasses__()

        for cls in classes:
            if cls.__column__ == name:
                return cls

        raise ValueError(f"Mapper {name} not found.")


class GenderMapper(BaseMapper):

    __column__ = "Gender"
    __data__ = {
                'Male':0,
                 "Female":1
                }
class FamilyMapper(BaseMapper):

    __column__='family_history_with_overweight'
    __data__={
                'yes':0,
                'no':1
    }
class FAVCMapper(BaseMapper):
    
    __column__='FAVC'
    __data__={
                'yes':0,
                'no':1
    }
class CAECMapper(BaseMapper):
    
    __column__='CAEC'
    __data__={
            'no':0, 
            'Sometimes':1, 
            'Frequently':2, 
            'Always':3
    }
class SMOKEMapper(BaseMapper):
    
    __column__='SMOKE'
    __data__={
            'yes':0,
            'no':1
    }
class SCCMapper(BaseMapper):
    
    __column__='SCC'
    __data__={
            'yes':0,
            'no':1
    }
class CALCMapper(BaseMapper):

    __column__='CALC'
    __data__={
            'no':0, 
            'Sometimes':1, 
            'Frequently':2, 
            'Always':3
    }
class MTRANSMapper(BaseMapper):
     
    __column__='MTRANS'
    __data__={
            'Automobile':0, 
            'Motorbike':1, 
            'Bike':2, 
            'Public_Transportation':3, 
            'Walking':4
    }
class NObeyMapper(BaseMapper):

    __column__='NObeyesdad'
    __data__={
            'Insufficient_Weight':0, 
            'Normal_Weight':1, 
            'Overweight_Level_I':2, 
            'Overweight_Level_II':3, 
            'Obesity_Type_I':4, 
            'Obesity_Type_II':5, 
            'Obesity_Type_III':6
    }
