########################################################
########################################################
## Main Data Setup:
########################################################
########################################################

### Topic Modeling
nohup sh -c 'for i in 20200101 20200102 20200103 20200104 20200105 20200106 20200107 20200108 20200109 20200110 20200111 20200112 20200113 20200114 20200115 20200116 20200117 20200118 20200119 20200120 20200121 20200122 20200123 20200124 20200125 20200126 20200127 20200128 20200129 20200130 20200131 20200201 20200202 20200203 20200204 20200205 20200206 20200207 20200208 20200209 20200210 20200211 20200212 20200213 20200214 20200215 20200216 20200217 20200218 20200219 20200220 20200221 20200222 20200223 20200224 20200225 20200226 20200227 20200228 20200229 20200301 20200302 20200303 20200304 20200305 20200306 20200307 20200308 20200309 20200310 20200311 20200312 20200313 20200314 20200315 20200316 20200317 20200318 20200319 20200320 20200321 20200322 20200323 20200324 20200325 20200326 20200327 20200328 20200329 20200330 20200331 20200401 20200402 20200403 20200404 20200405 20200406 20200407 20200408 20200409 20200410 20200411 20200412 20200413 20200414 20200415 20200416 20200417 20200418 20200419 20200420 20200421 20200422 20200423 20200424 20200425 20200426 20200427 20200428 20200429 20200430; do python3 -u src/py/topic_modeling.py $i; done > topic_modeling.log' &

########################################################
########################################################
## Subsetting data to topic:
########################################################
########################################################
### For COVID:
nohup python3 -u src/twitter_data_extraction.py covid --incl_keywords coronavirus covid covid19 --country USA --lang en --start_date 2020-01-01 --end_date 2020-09-30 --text_path /home/sentiment/data-lake/twitter/processed/ --geo_path /home/sentiment/data-lake/twitter/geoinfo/ > covid_extraction.log &

### For Chinese Virus:
nohup python3 -u src/twitter_data_extraction.py china_virus --incl_keywords chinese\s?virus china\s?virus chinese\s?coronavirus china\s?coronavirus --country USA --lang en --start_date 2020-01-01 --end_date 2020-09-30 --text_path /home/sentiment/data-lake/twitter/processed/ --geo_path /home/sentiment/data-lake/twitter/geoinfo/ > china_virus_extraction.log &

### For Climate Change in Portugal
nohup python3 -u src/twitter_data_extraction.py climate_change_prt --incl_keywords climate\s?change global\s?warming --country PRT --lang en --start_date 2019-01-01 --end_date 2020-09-30 --text_path /home/sentiment/data-lake/twitter/processed/ --geo_path /home/sentiment/data-lake/twitter/geoinfo/ > climate_change_prt_extraction.log &
