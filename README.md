# Run using justfile

Run the finetuning pipeline: `just run` 

# Benchmark Scores
- `1B`: 0.835273989136995 `meta-llama/Llama-3.2-1B-Instruct`
- `3B`: 0.8572563705583769 `meta-llama/Llama-3.2-3B-Instruct`
- `8B`: 0.8534518351496928 `meta-llama/Meta-Llama-3-8B-Instruct`


## Error Analysis
### 1B Model
```json
[
  {
    "Question no": 1,
    "difference": 0.1924
  },
  {
    "Question no": 2,
    "difference": 0.1209
  },
  {
    "Question no": 3,
    "difference": 0.1679
  },
  {
    "Question no": 4,
    "difference": 0.1891
  },
  {
    "Question no": 5,
    "difference": 0.1259
  },
  {
    "Question no": 6,
    "difference": 0.1684
  },
  {
    "Question no": 7,
    "difference": 0.2071
  },
  {
    "Question no": 8,
    "difference": 0.1288
  },
  {
    "Question no": 9,
    "difference": 0.1768
  },
  {
    "Question no": 10,
    "difference": 0.1694
  },
  {
    "Question no": 11,
    "difference": 0.1502
  },
  {
    "Question no": 12,
    "difference": 0.2024
  },
  {
    "Question no": 13,
    "difference": 0.2179
  },
  {
    "Question no": 14,
    "difference": 0.1526
  },
  {
    "Question no": 15,
    "difference": 0.1758
  },
  {
    "Question no": 16,
    "difference": 0.1703
  },
  {
    "Question no": 17,
    "difference": 0.144
  },
  {
    "Question no": 18,
    "difference": 0.1624
  },
  {
    "Question no": 19,
    "difference": 0.19
  },
  {
    "Question no": 20,
    "difference": 0.2146
  },
  {
    "Question no": 21,
    "difference": 0.1139
  },
  {
    "Question no": 22,
    "difference": 0.1762
  },
  {
    "Question no": 23,
    "difference": 0.1399
  },
  {
    "Question no": 24,
    "difference": 0.1661
  },
  {
    "Question no": 25,
    "difference": 0.1775
  },
  {
    "Question no": 26,
    "difference": 0.2003
  },
  {
    "Question no": 27,
    "difference": 0.0975
  },
  {
    "Question no": 28,
    "difference": 0.1135
  },
  {
    "Question no": 29,
    "difference": 0.1595
  },
  {
    "Question no": 30,
    "difference": 0.1084
  },
  {
    "Question no": 31,
    "difference": 0.166
  },
  {
    "Question no": 32,
    "difference": 0.1486
  },
  {
    "Question no": 33,
    "difference": 0.0764
  },
  {
    "Question no": 34,
    "difference": 0.1599
  },
  {
    "Question no": 35,
    "difference": 0.1392
  },
  {
    "Question no": 36,
    "difference": 0.094
  },
  {
    "Question no": 37,
    "difference": 0.1015
  },
  {
    "Question no": 38,
    "difference": 0.1801
  },
  {
    "Question no": 39,
    "difference": 0.1552
  },
  {
    "Question no": 40,
    "difference": 0.0832
  },
  {
    "Question no": 41,
    "difference": 0.1884
  },
  {
    "Question no": 42,
    "difference": 0.1485
  },
  {
    "Question no": 43,
    "difference": 0.1217
  },
  {
    "Question no": 44,
    "difference": 0.1196
  },
  {
    "Question no": 45,
    "difference": 0.1772
  },
  {
    "Question no": 46,
    "difference": 0.1761
  },
  {
    "Question no": 47,
    "difference": 0.1211
  },
  {
    "Question no": 48,
    "difference": 0.1573
  },
  {
    "Question no": 49,
    "difference": 0.1841
  },
  {
    "Question no": 50,
    "difference": 0.1658
  },
  {
    "Question no": 51,
    "difference": 0.14
  },
  {
    "Question no": 52,
    "difference": 0.1542
  },
  {
    "Question no": 53,
    "difference": 0.147
  },
  {
    "Question no": 54,
    "difference": 0.183
  },
  {
    "Question no": 55,
    "difference": 0.1167
  },
  {
    "Question no": 56,
    "difference": 0.1566
  },
  {
    "Question no": 57,
    "difference": 0.1412
  },
  {
    "Question no": 58,
    "difference": 0.1457
  },
  {
    "Question no": 59,
    "difference": 0.1767
  },
  {
    "Question no": 60,
    "difference": 0.1098
  },
  {
    "Question no": 61,
    "difference": 0.0994
  },
  {
    "Question no": 62,
    "difference": 0.2077
  },
  {
    "Question no": 63,
    "difference": 0.182
  },
  {
    "Question no": 64,
    "difference": 0.1368
  },
  {
    "Question no": 65,
    "difference": 0.0875
  },
  {
    "Question no": 66,
    "difference": 0.1733
  },
  {
    "Question no": 67,
    "difference": 0.1203
  },
  {
    "Question no": 68,
    "difference": 0.1136
  },
  {
    "Question no": 69,
    "difference": 0.1285
  },
  {
    "Question no": 70,
    "difference": 0.1127
  },
  {
    "Question no": 71,
    "difference": 0.1804
  },
  {
    "Question no": 72,
    "difference": 0.0527
  },
  {
    "Question no": 73,
    "difference": 0.2138
  },
  {
    "Question no": 74,
    "difference": 0.112
  },
  {
    "Question no": 75,
    "difference": 0.1359
  },
  {
    "Question no": 76,
    "difference": 0.1047
  },
  {
    "Question no": 77,
    "difference": 0.1855
  },
  {
    "Question no": 78,
    "difference": 0.1583
  },
  {
    "Question no": 79,
    "difference": 0.1864
  },
  {
    "Question no": 80,
    "difference": 0.1833
  },
  {
    "Question no": 81,
    "difference": 0.1774
  },
  {
    "Question no": 82,
    "difference": 0.1233
  },
  {
    "Question no": 83,
    "difference": 0.2276
  },
  {
    "Question no": 84,
    "difference": 0.174
  },
  {
    "Question no": 85,
    "difference": 0.149
  },
  {
    "Question no": 86,
    "difference": 0.1626
  },
  {
    "Question no": 87,
    "difference": 0.1986
  },
  {
    "Question no": 88,
    "difference": 0.2039
  },
  {
    "Question no": 89,
    "difference": 0.1667
  },
  {
    "Question no": 90,
    "difference": 0.1652
  },
  {
    "Question no": 91,
    "difference": 0.2039
  },
  {
    "Question no": 92,
    "difference": 0.0631
  },
  {
    "Question no": 93,
    "difference": 0.1646
  },
  {
    "Question no": 94,
    "difference": 0.1806
  },
  {
    "Question no": 95,
    "difference": 0.1526
  },
  {
    "Question no": 96,
    "difference": 0.1482
  },
  {
    "Question no": 97,
    "difference": 0.1958
  },
  {
    "Question no": 98,
    "difference": 0.1851
  },
  {
    "Question no": 99,
    "difference": 0.2199
  },
  {
    "Question no": 100,
    "difference": 0.0816
  },
  {
    "Question no": 101,
    "difference": 0.1498
  },
  {
    "Question no": 102,
    "difference": 0.0863
  },
  {
    "Question no": 103,
    "difference": 0.1397
  },
  {
    "Question no": 104,
    "difference": 0.2077
  },
  {
    "Question no": 105,
    "difference": 0.16
  },
  {
    "Question no": 106,
    "difference": 0.1423
  },
  {
    "Question no": 107,
    "difference": 0.2048
  },
  {
    "Question no": 108,
    "difference": 0.1348
  },
  {
    "Question no": 109,
    "difference": 0.185
  },
  {
    "Question no": 110,
    "difference": 0.1483
  },
  {
    "Question no": 111,
    "difference": 0.1029
  },
  {
    "Question no": 112,
    "difference": 0.1826
  },
  {
    "Question no": 113,
    "difference": 0.1264
  },
  {
    "Question no": 114,
    "difference": 0.1575
  },
  {
    "Question no": 115,
    "difference": 0.1044
  },
  {
    "Question no": 116,
    "difference": 0.0706
  },
  {
    "Question no": 117,
    "difference": 0.121
  },
  {
    "Question no": 118,
    "difference": 0.2039
  },
  {
    "Question no": 119,
    "difference": 0.1893
  },
  {
    "Question no": 120,
    "difference": 0.1686
  },
  {
    "Question no": 121,
    "difference": 0.1601
  },
  {
    "Question no": 122,
    "difference": 0.1794
  },
  {
    "Question no": 123,
    "difference": 0.1739
  },
  {
    "Question no": 124,
    "difference": 0.1892
  },
  {
    "Question no": 125,
    "difference": 0.1064
  },
  {
    "Question no": 126,
    "difference": 0.1142
  },
  {
    "Question no": 127,
    "difference": 0.1563
  },
  {
    "Question no": 128,
    "difference": 0.1914
  },
  {
    "Question no": 129,
    "difference": 0.1594
  },
  {
    "Question no": 130,
    "difference": 0.1565
  },
  {
    "Question no": 131,
    "difference": 0.1906
  },
  {
    "Question no": 132,
    "difference": 0.1456
  },
  {
    "Question no": 133,
    "difference": 0.115
  },
  {
    "Question no": 134,
    "difference": 0.1964
  },
  {
    "Question no": 135,
    "difference": 0.0973
  },
  {
    "Question no": 136,
    "difference": 0.1711
  },
  {
    "Question no": 137,
    "difference": 0.1843
  },
  {
    "Question no": 138,
    "difference": 0.1419
  },
  {
    "Question no": 139,
    "difference": 0.1892
  },
  {
    "Question no": 140,
    "difference": 0.1102
  },
  {
    "Question no": 141,
    "difference": 0.1695
  },
  {
    "Question no": 142,
    "difference": 0.156
  },
  {
    "Question no": 143,
    "difference": 0.1579
  },
  {
    "Question no": 144,
    "difference": 0.0982
  },
  {
    "Question no": 145,
    "difference": 0.1905
  },
  {
    "Question no": 146,
    "difference": 0.1227
  },
  {
    "Question no": 147,
    "difference": 0.1898
  },
  {
    "Question no": 148,
    "difference": 0.0836
  },
  {
    "Question no": 149,
    "difference": 0.1121
  },
  {
    "Question no": 150,
    "difference": 0.1073
  },
  {
    "Question no": 151,
    "difference": 0.178
  },
  {
    "Question no": 152,
    "difference": 0.1959
  },
  {
    "Question no": 153,
    "difference": 0.1722
  },
  {
    "Question no": 154,
    "difference": 0.1431
  },
  {
    "Question no": 155,
    "difference": 0.2034
  },
  {
    "Question no": 156,
    "difference": 0.1601
  },
  {
    "Question no": 157,
    "difference": 0.1918
  },
  {
    "Question no": 158,
    "difference": 0.1573
  },
  {
    "Question no": 159,
    "difference": 0.2065
  },
  {
    "Question no": 160,
    "difference": 0.111
  },
  {
    "Question no": 161,
    "difference": 0.1689
  },
  {
    "Question no": 162,
    "difference": 0.1574
  },
  {
    "Question no": 163,
    "difference": 0.1247
  },
  {
    "Question no": 164,
    "difference": 0.1053
  },
  {
    "Question no": 165,
    "difference": 0.0919
  },
  {
    "Question no": 166,
    "difference": 0.1556
  },
  {
    "Question no": 167,
    "difference": 0.158
  },
  {
    "Question no": 168,
    "difference": 0.1221
  },
  {
    "Question no": 169,
    "difference": 0.0986
  },
  {
    "Question no": 170,
    "difference": 0.1707
  }
]
```

### 3B Model
```json
[
  {
    "Question no": 1,
    "difference": 0.1023
  },
  {
    "Question no": 2,
    "difference": 0.1153
  },
  {
    "Question no": 3,
    "difference": 0.1394
  },
  {
    "Question no": 4,
    "difference": 0.1229
  },
  {
    "Question no": 5,
    "difference": 0.1049
  },
  {
    "Question no": 6,
    "difference": 0.1105
  },
  {
    "Question no": 7,
    "difference": 0.1293
  },
  {
    "Question no": 8,
    "difference": 0.1283
  },
  {
    "Question no": 9,
    "difference": 0.1196
  },
  {
    "Question no": 10,
    "difference": 0.0916
  },
  {
    "Question no": 11,
    "difference": 0.1109
  },
  {
    "Question no": 12,
    "difference": 0.1078
  },
  {
    "Question no": 13,
    "difference": 0.1061
  },
  {
    "Question no": 14,
    "difference": 0.1104
  },
  {
    "Question no": 15,
    "difference": 0.1372
  },
  {
    "Question no": 16,
    "difference": 0.1434
  },
  {
    "Question no": 17,
    "difference": 0.0828
  },
  {
    "Question no": 18,
    "difference": 0.1284
  },
  {
    "Question no": 19,
    "difference": 0.218
  },
  {
    "Question no": 20,
    "difference": 0.1484
  },
  {
    "Question no": 21,
    "difference": 0.2286
  },
  {
    "Question no": 22,
    "difference": 0.1939
  },
  {
    "Question no": 23,
    "difference": 0.1407
  },
  {
    "Question no": 24,
    "difference": 0.1879
  },
  {
    "Question no": 25,
    "difference": 0.17
  },
  {
    "Question no": 26,
    "difference": 0.1809
  },
  {
    "Question no": 27,
    "difference": 0.1181
  },
  {
    "Question no": 28,
    "difference": 0.0629
  },
  {
    "Question no": 29,
    "difference": 0.1145
  },
  {
    "Question no": 30,
    "difference": 0.1455
  },
  {
    "Question no": 31,
    "difference": 0.2011
  },
  {
    "Question no": 32,
    "difference": 0.1238
  },
  {
    "Question no": 33,
    "difference": 0.1072
  },
  {
    "Question no": 34,
    "difference": 0.1425
  },
  {
    "Question no": 35,
    "difference": 0.115
  },
  {
    "Question no": 36,
    "difference": 0.1405
  },
  {
    "Question no": 37,
    "difference": 0.1793
  },
  {
    "Question no": 38,
    "difference": 0.2107
  },
  {
    "Question no": 39,
    "difference": 0.0876
  },
  {
    "Question no": 40,
    "difference": 0.1254
  },
  {
    "Question no": 41,
    "difference": 0.1769
  },
  {
    "Question no": 42,
    "difference": 0.1594
  },
  {
    "Question no": 43,
    "difference": 0.2072
  },
  {
    "Question no": 44,
    "difference": 0.1974
  },
  {
    "Question no": 45,
    "difference": 0.1855
  },
  {
    "Question no": 46,
    "difference": 0.1903
  },
  {
    "Question no": 47,
    "difference": 0.0717
  },
  {
    "Question no": 48,
    "difference": 0.1037
  },
  {
    "Question no": 49,
    "difference": 0.0979
  },
  {
    "Question no": 50,
    "difference": 0.0907
  },
  {
    "Question no": 51,
    "difference": 0.091
  },
  {
    "Question no": 52,
    "difference": 0.1103
  },
  {
    "Question no": 53,
    "difference": 0.1828
  },
  {
    "Question no": 54,
    "difference": 0.1143
  },
  {
    "Question no": 55,
    "difference": 0.1226
  },
  {
    "Question no": 56,
    "difference": 0.0897
  },
  {
    "Question no": 57,
    "difference": 0.1415
  },
  {
    "Question no": 58,
    "difference": 0.1473
  },
  {
    "Question no": 59,
    "difference": 0.1391
  },
  {
    "Question no": 60,
    "difference": 0.1586
  },
  {
    "Question no": 61,
    "difference": 0.2143
  },
  {
    "Question no": 62,
    "difference": 0.2066
  },
  {
    "Question no": 63,
    "difference": 0.1165
  },
  {
    "Question no": 64,
    "difference": 0.1733
  },
  {
    "Question no": 65,
    "difference": 0.1407
  },
  {
    "Question no": 66,
    "difference": 0.1491
  },
  {
    "Question no": 67,
    "difference": 0.1489
  },
  {
    "Question no": 68,
    "difference": 0.205
  },
  {
    "Question no": 69,
    "difference": 0.1282
  },
  {
    "Question no": 70,
    "difference": 0.1419
  },
  {
    "Question no": 71,
    "difference": 0.1063
  },
  {
    "Question no": 72,
    "difference": 0.1635
  },
  {
    "Question no": 73,
    "difference": 0.1905
  },
  {
    "Question no": 74,
    "difference": 0.0938
  },
  {
    "Question no": 75,
    "difference": 0.0965
  },
  {
    "Question no": 76,
    "difference": 0.1555
  },
  {
    "Question no": 77,
    "difference": 0.0984
  },
  {
    "Question no": 78,
    "difference": 0.1487
  },
  {
    "Question no": 79,
    "difference": 0.1431
  },
  {
    "Question no": 80,
    "difference": 0.1744
  },
  {
    "Question no": 81,
    "difference": 0.1451
  },
  {
    "Question no": 82,
    "difference": 0.1257
  },
  {
    "Question no": 83,
    "difference": 0.1689
  },
  {
    "Question no": 84,
    "difference": 0.1008
  },
  {
    "Question no": 85,
    "difference": 0.1658
  },
  {
    "Question no": 86,
    "difference": 0.1288
  },
  {
    "Question no": 87,
    "difference": 0.2013
  },
  {
    "Question no": 88,
    "difference": 0.1272
  },
  {
    "Question no": 89,
    "difference": 0.1805
  },
  {
    "Question no": 90,
    "difference": 0.1483
  },
  {
    "Question no": 91,
    "difference": 0.1588
  },
  {
    "Question no": 92,
    "difference": 0.1725
  },
  {
    "Question no": 93,
    "difference": 0.1466
  },
  {
    "Question no": 94,
    "difference": 0.1129
  },
  {
    "Question no": 95,
    "difference": 0.1777
  },
  {
    "Question no": 96,
    "difference": 0.0831
  },
  {
    "Question no": 97,
    "difference": 0.107
  },
  {
    "Question no": 98,
    "difference": 0.0457
  },
  {
    "Question no": 99,
    "difference": 0.152
  },
  {
    "Question no": 100,
    "difference": 0.1472
  },
  {
    "Question no": 101,
    "difference": 0.1322
  },
  {
    "Question no": 102,
    "difference": 0.0923
  },
  {
    "Question no": 103,
    "difference": 0.0741
  },
  {
    "Question no": 104,
    "difference": 0.1363
  },
  {
    "Question no": 105,
    "difference": 0.1568
  },
  {
    "Question no": 106,
    "difference": 0.1874
  },
  {
    "Question no": 107,
    "difference": 0.1806
  },
  {
    "Question no": 108,
    "difference": 0.0791
  },
  {
    "Question no": 109,
    "difference": 0.1986
  },
  {
    "Question no": 110,
    "difference": 0.2072
  },
  {
    "Question no": 111,
    "difference": 0.1313
  },
  {
    "Question no": 112,
    "difference": 0.1093
  },
  {
    "Question no": 113,
    "difference": 0.1038
  },
  {
    "Question no": 114,
    "difference": 0.0832
  },
  {
    "Question no": 115,
    "difference": 0.1367
  },
  {
    "Question no": 116,
    "difference": 0.0963
  },
  {
    "Question no": 117,
    "difference": 0.183
  },
  {
    "Question no": 118,
    "difference": 0.1403
  },
  {
    "Question no": 119,
    "difference": 0.1262
  },
  {
    "Question no": 120,
    "difference": 0.1164
  },
  {
    "Question no": 121,
    "difference": 0.1381
  },
  {
    "Question no": 122,
    "difference": 0.0808
  },
  {
    "Question no": 123,
    "difference": 0.1469
  },
  {
    "Question no": 124,
    "difference": 0.0639
  },
  {
    "Question no": 125,
    "difference": 0.0301
  },
  {
    "Question no": 126,
    "difference": 0.1357
  },
  {
    "Question no": 127,
    "difference": 0.1661
  },
  {
    "Question no": 128,
    "difference": 0.0946
  },
  {
    "Question no": 129,
    "difference": 0.1336
  },
  {
    "Question no": 130,
    "difference": 0.1447
  },
  {
    "Question no": 131,
    "difference": 0.1833
  },
  {
    "Question no": 132,
    "difference": 0.0913
  },
  {
    "Question no": 133,
    "difference": 0.1661
  },
  {
    "Question no": 134,
    "difference": 0.1461
  },
  {
    "Question no": 135,
    "difference": 0.0771
  },
  {
    "Question no": 136,
    "difference": 0.1746
  },
  {
    "Question no": 137,
    "difference": 0.1388
  },
  {
    "Question no": 138,
    "difference": 0.0878
  },
  {
    "Question no": 139,
    "difference": 0.0816
  },
  {
    "Question no": 140,
    "difference": 0.164
  },
  {
    "Question no": 141,
    "difference": 0.1297
  },
  {
    "Question no": 142,
    "difference": 0.1373
  },
  {
    "Question no": 143,
    "difference": 0.1724
  },
  {
    "Question no": 144,
    "difference": 0.1555
  },
  {
    "Question no": 145,
    "difference": 0.201
  },
  {
    "Question no": 146,
    "difference": 0.1901
  },
  {
    "Question no": 147,
    "difference": 0.1797
  },
  {
    "Question no": 148,
    "difference": 0.1751
  },
  {
    "Question no": 149,
    "difference": 0.1508
  },
  {
    "Question no": 150,
    "difference": 0.0994
  },
  {
    "Question no": 151,
    "difference": 0.0857
  },
  {
    "Question no": 152,
    "difference": 0.0642
  },
  {
    "Question no": 153,
    "difference": 0.1239
  },
  {
    "Question no": 154,
    "difference": 0.0874
  },
  {
    "Question no": 155,
    "difference": 0.1074
  },
  {
    "Question no": 156,
    "difference": 0.0944
  },
  {
    "Question no": 157,
    "difference": 0.1194
  },
  {
    "Question no": 158,
    "difference": 0.1553
  },
  {
    "Question no": 159,
    "difference": 0.1041
  },
  {
    "Question no": 160,
    "difference": 0.1534
  },
  {
    "Question no": 161,
    "difference": 0.1637
  }
]
```

### 8B Model
```json
[
  {
    "Question no": 1,
    "difference": 0.1601
  },
  {
    "Question no": 2,
    "difference": 0.1673
  },
  {
    "Question no": 3,
    "difference": 0.1749
  },
  {
    "Question no": 4,
    "difference": 0.1201
  },
  {
    "Question no": 5,
    "difference": 0.1101
  },
  {
    "Question no": 6,
    "difference": 0.1227
  },
  {
    "Question no": 7,
    "difference": 0.1172
  },
  {
    "Question no": 8,
    "difference": 0.1727
  },
  {
    "Question no": 9,
    "difference": 0.1987
  },
  {
    "Question no": 10,
    "difference": 0.1935
  },
  {
    "Question no": 11,
    "difference": 0.2233
  },
  {
    "Question no": 12,
    "difference": 0.1936
  },
  {
    "Question no": 13,
    "difference": 0.2064
  },
  {
    "Question no": 14,
    "difference": 0.1739
  },
  {
    "Question no": 15,
    "difference": 0.138
  },
  {
    "Question no": 16,
    "difference": 0.1308
  },
  {
    "Question no": 17,
    "difference": 0.2154
  },
  {
    "Question no": 18,
    "difference": 0.1248
  },
  {
    "Question no": 19,
    "difference": 0.155
  },
  {
    "Question no": 20,
    "difference": 0.201
  },
  {
    "Question no": 21,
    "difference": 0.132
  },
  {
    "Question no": 22,
    "difference": 0.1841
  },
  {
    "Question no": 23,
    "difference": 0.2092
  },
  {
    "Question no": 24,
    "difference": 0.1277
  },
  {
    "Question no": 25,
    "difference": 0.1072
  },
  {
    "Question no": 26,
    "difference": 0.0939
  },
  {
    "Question no": 27,
    "difference": 0.1945
  },
  {
    "Question no": 28,
    "difference": 0.1945
  },
  {
    "Question no": 29,
    "difference": 0.0727
  },
  {
    "Question no": 30,
    "difference": 0.1218
  },
  {
    "Question no": 31,
    "difference": 0.1511
  },
  {
    "Question no": 32,
    "difference": 0.1621
  },
  {
    "Question no": 33,
    "difference": 0.1463
  },
  {
    "Question no": 34,
    "difference": 0.1383
  },
  {
    "Question no": 35,
    "difference": 0.1624
  },
  {
    "Question no": 36,
    "difference": 0.0844
  },
  {
    "Question no": 37,
    "difference": 0.1097
  },
  {
    "Question no": 38,
    "difference": 0.1373
  },
  {
    "Question no": 39,
    "difference": 0.1574
  },
  {
    "Question no": 40,
    "difference": 0.0726
  },
  {
    "Question no": 41,
    "difference": 0.1081
  },
  {
    "Question no": 42,
    "difference": 0.1785
  },
  {
    "Question no": 43,
    "difference": 0.1349
  },
  {
    "Question no": 44,
    "difference": 0.1634
  },
  {
    "Question no": 45,
    "difference": 0.0869
  },
  {
    "Question no": 46,
    "difference": 0.1233
  },
  {
    "Question no": 47,
    "difference": 0.1407
  },
  {
    "Question no": 48,
    "difference": 0.2087
  },
  {
    "Question no": 49,
    "difference": 0.1913
  },
  {
    "Question no": 50,
    "difference": 0.2
  },
  {
    "Question no": 51,
    "difference": 0.1574
  },
  {
    "Question no": 52,
    "difference": 0.0937
  },
  {
    "Question no": 53,
    "difference": 0.1463
  },
  {
    "Question no": 54,
    "difference": 0.1205
  },
  {
    "Question no": 55,
    "difference": 0.1177
  },
  {
    "Question no": 56,
    "difference": 0.1681
  },
  {
    "Question no": 57,
    "difference": 0.195
  },
  {
    "Question no": 58,
    "difference": 0.139
  },
  {
    "Question no": 59,
    "difference": 0.1545
  },
  {
    "Question no": 60,
    "difference": 0.1659
  },
  {
    "Question no": 61,
    "difference": 0.2092
  },
  {
    "Question no": 62,
    "difference": 0.2048
  },
  {
    "Question no": 63,
    "difference": 0.2121
  },
  {
    "Question no": 64,
    "difference": 0.17
  },
  {
    "Question no": 65,
    "difference": 0.1908
  },
  {
    "Question no": 66,
    "difference": 0.2359
  },
  {
    "Question no": 67,
    "difference": 0.1376
  },
  {
    "Question no": 68,
    "difference": 0.1672
  },
  {
    "Question no": 69,
    "difference": 0.1912
  },
  {
    "Question no": 70,
    "difference": 0.0908
  },
  {
    "Question no": 71,
    "difference": 0.1187
  },
  {
    "Question no": 72,
    "difference": 0.1562
  },
  {
    "Question no": 73,
    "difference": 0.1426
  },
  {
    "Question no": 74,
    "difference": 0.2037
  },
  {
    "Question no": 75,
    "difference": 0.1868
  },
  {
    "Question no": 76,
    "difference": 0.042
  },
  {
    "Question no": 77,
    "difference": 0.2019
  },
  {
    "Question no": 78,
    "difference": 0.1599
  },
  {
    "Question no": 79,
    "difference": 0.2112
  },
  {
    "Question no": 80,
    "difference": 0.1467
  },
  {
    "Question no": 81,
    "difference": 0.1503
  },
  {
    "Question no": 82,
    "difference": 0.1287
  },
  {
    "Question no": 83,
    "difference": 0.1485
  },
  {
    "Question no": 84,
    "difference": 0.1878
  },
  {
    "Question no": 85,
    "difference": 0.1465
  },
  {
    "Question no": 86,
    "difference": 0.1265
  },
  {
    "Question no": 87,
    "difference": 0.2007
  },
  {
    "Question no": 88,
    "difference": 0.1929
  },
  {
    "Question no": 89,
    "difference": 0.2154
  },
  {
    "Question no": 90,
    "difference": 0.1342
  },
  {
    "Question no": 91,
    "difference": 0.0782
  },
  {
    "Question no": 92,
    "difference": 0.1411
  },
  {
    "Question no": 93,
    "difference": 0.1672
  },
  {
    "Question no": 94,
    "difference": 0.1492
  },
  {
    "Question no": 95,
    "difference": 0.1604
  },
  {
    "Question no": 96,
    "difference": 0.2163
  },
  {
    "Question no": 97,
    "difference": 0.1549
  },
  {
    "Question no": 98,
    "difference": 0.1034
  },
  {
    "Question no": 99,
    "difference": 0.1358
  },
  {
    "Question no": 100,
    "difference": 0.1726
  },
  {
    "Question no": 101,
    "difference": 0.2082
  },
  {
    "Question no": 102,
    "difference": 0.2123
  },
  {
    "Question no": 103,
    "difference": 0.2029
  },
  {
    "Question no": 104,
    "difference": 0.2046
  },
  {
    "Question no": 105,
    "difference": 0.1828
  },
  {
    "Question no": 106,
    "difference": 0.2165
  },
  {
    "Question no": 107,
    "difference": 0.2133
  },
  {
    "Question no": 108,
    "difference": 0.1692
  },
  {
    "Question no": 109,
    "difference": 0.1586
  },
  {
    "Question no": 110,
    "difference": 0.1458
  },
  {
    "Question no": 111,
    "difference": 0.0979
  },
  {
    "Question no": 112,
    "difference": 0.0974
  },
  {
    "Question no": 113,
    "difference": 0.0756
  },
  {
    "Question no": 114,
    "difference": 0.1823
  },
  {
    "Question no": 115,
    "difference": 0.2121
  },
  {
    "Question no": 116,
    "difference": 0.1369
  },
  {
    "Question no": 117,
    "difference": 0.0914
  },
  {
    "Question no": 118,
    "difference": 0.0999
  },
  {
    "Question no": 119,
    "difference": 0.1111
  },
  {
    "Question no": 120,
    "difference": 0.0448
  },
  {
    "Question no": 121,
    "difference": 0.1426
  },
  {
    "Question no": 122,
    "difference": 0.1371
  },
  {
    "Question no": 123,
    "difference": 0.204
  },
  {
    "Question no": 124,
    "difference": 0.1408
  },
  {
    "Question no": 125,
    "difference": 0.1593
  },
  {
    "Question no": 126,
    "difference": 0.1983
  },
  {
    "Question no": 127,
    "difference": 0.1384
  },
  {
    "Question no": 128,
    "difference": 0.217
  },
  {
    "Question no": 129,
    "difference": 0.1486
  },
  {
    "Question no": 130,
    "difference": 0.122
  },
  {
    "Question no": 131,
    "difference": 0.203
  },
  {
    "Question no": 132,
    "difference": 0.1391
  },
  {
    "Question no": 133,
    "difference": 0.1783
  },
  {
    "Question no": 134,
    "difference": 0.1019
  },
  {
    "Question no": 135,
    "difference": 0.1316
  },
  {
    "Question no": 136,
    "difference": 0.1391
  },
  {
    "Question no": 137,
    "difference": 0.0991
  },
  {
    "Question no": 138,
    "difference": 0.1199
  },
  {
    "Question no": 139,
    "difference": 0.1183
  },
  {
    "Question no": 140,
    "difference": 0.1327
  },
  {
    "Question no": 141,
    "difference": 0.1447
  },
  {
    "Question no": 142,
    "difference": 0.1568
  },
  {
    "Question no": 143,
    "difference": 0.1367
  },
  {
    "Question no": 144,
    "difference": 0.1871
  },
  {
    "Question no": 145,
    "difference": 0.138
  },
  {
    "Question no": 146,
    "difference": 0.1845
  },
  {
    "Question no": 147,
    "difference": 0.1213
  },
  {
    "Question no": 148,
    "difference": 0.1823
  },
  {
    "Question no": 149,
    "difference": 0.1239
  },
  {
    "Question no": 150,
    "difference": 0.1632
  },
  {
    "Question no": 151,
    "difference": 0.0765
  },
  {
    "Question no": 152,
    "difference": 0.1969
  },
  {
    "Question no": 153,
    "difference": 0.21
  },
  {
    "Question no": 154,
    "difference": 0.1719
  },
  {
    "Question no": 155,
    "difference": 0.1195
  },
  {
    "Question no": 156,
    "difference": 0.197
  },
  {
    "Question no": 157,
    "difference": 0.1769
  },
  {
    "Question no": 158,
    "difference": 0.1984
  },
  {
    "Question no": 159,
    "difference": 0.1736
  },
  {
    "Question no": 160,
    "difference": 0.1681
  },
  {
    "Question no": 161,
    "difference": 0.176
  },
  {
    "Question no": 162,
    "difference": 0.196
  },
  {
    "Question no": 163,
    "difference": 0.1282
  },
  {
    "Question no": 164,
    "difference": 0.1704
  },
  {
    "Question no": 165,
    "difference": 0.1712
  },
  {
    "Question no": 166,
    "difference": 0.1611
  },
  {
    "Question no": 167,
    "difference": 0.1217
  },
  {
    "Question no": 168,
    "difference": 0.035
  },
  {
    "Question no": 169,
    "difference": 0.1043
  },
  {
    "Question no": 170,
    "difference": 0.1277
  },
  {
    "Question no": 171,
    "difference": 0.111
  },
  {
    "Question no": 172,
    "difference": 0.133
  },
  {
    "Question no": 173,
    "difference": 0.1099
  },
  {
    "Question no": 174,
    "difference": 0.1444
  },
  {
    "Question no": 175,
    "difference": 0.1279
  },
  {
    "Question no": 176,
    "difference": 0.1785
  },
  {
    "Question no": 177,
    "difference": 0.1545
  },
  {
    "Question no": 178,
    "difference": 0.1054
  },
  {
    "Question no": 179,
    "difference": 0.1113
  },
  {
    "Question no": 180,
    "difference": 0.1382
  }
]

```