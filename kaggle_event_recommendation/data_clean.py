# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/5/10 14:13
@Function:
"""
import datetime
import hashlib
from collections import defaultdict
import locale

import pycountry


class DataCleaner:
    """
    Common utilities for converting strings to equivalent numbers
    or number buckets.
    """

    def __init__(self):
        # 载入 locales
        self.localeIdMap = defaultdict(int)
        for i, l in enumerate(locale.locale_alias.keys()):
            self.localeIdMap[l] = i + 1
        # 载入 countries
        self.countryIdMap = defaultdict(int)
        ctryIdx = defaultdict(int)
        for i, c in enumerate(pycountry.countries):
            self.countryIdMap[c.name.lower()] = i + 1
            if c.name.lower() == "usa":
                ctryIdx["US"] = i
            if c.name.lower() == "canada":
                ctryIdx["CA"] = i
        for cc in ctryIdx.keys():
            for s in pycountry.subdivisions.get(country_code=cc):
                self.countryIdMap[s.name.lower()] = ctryIdx[cc] + 1
        # 载入 gender id 字典
        self.genderIdMap = defaultdict(int, {"male": 1, "female": 2})

    def getLocaleId(self, locstr):
        return self.localeIdMap[locstr.lower()]

    def getGenderId(self, genderStr):
        return self.genderIdMap[genderStr]

    def getJoinedYearMonth(self, dateString):
        dttm = datetime.datetime.strptime(dateString, "%Y-%m-%dT%H:%M:%S.%fZ")
        return "".join([str(dttm.year), str(dttm.month)])

    def getCountryId(self, location):
        if (isinstance(location, str)
                and len(location.strip()) > 0
                and location.rfind("  ") > -1):
            return self.countryIdMap[location[location.rindex("  ") + 2:].lower()]
        else:
            return 0

    def getBirthYearInt(self, birthYear):
        try:
            return 0 if birthYear == "None" else int(birthYear)
        except:
            return 0

    def getTimezoneInt(self, timezone):
        try:
            return int(timezone)
        except:
            return 0

    def getFeatureHash(self, value):
        if len(value.strip()) == 0:
            return -1
        else:
            return int(hashlib.sha224(value.encode('utf-8')).hexdigest()[0:4], 16)

    def getFloatValue(self, value):
        if len(value.strip()) == 0:
            return 0.0
        else:
            return float(value)
