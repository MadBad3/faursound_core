# faursound_core
[![pytest](https://github.com/GVSCAL/faursound_core/actions/workflows/pytest.yml/badge.svg)](https://github.com/GVSCAL/faursound_core/actions/workflows/pytest.yml)
[![CodeQL](https://github.com/GVSCAL/faursound_core/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/GVSCAL/faursound_core/actions/workflows/codeql-analysis.yml)
[![Docker Image CI](https://github.com/GVSCAL/faursound_core/actions/workflows/docker-image-with-test.yml/badge.svg)](https://github.com/GVSCAL/faursound_core/actions/workflows/docker-image-with-test.yml)
## Install
```python 
conda create --name fs python=3.8
```
```python 
conda activate fs
```
```python
pip install -r requirements.txt
```
note : if issue when installing elastic-apm -> download source file, then install manually with
```python
python setup.py install
```

## FaurSound core functions

see doc, link TBU

## FaurSound REST API

host API server with :

```python
uvicorn mainEOL:app
```
check API docs in: http://localhost:8000/docs

![image-20210629090554335](README.assets/image-20210629090554335.png)

## Monitoring API performance

**Note before start : this set up need 17G+ free RAM in your computer !**

High-Level Architecture :

![image-20210629223818749](README.assets/image-20210629223818749.png)

elasticsearch download & how to , see : https://www.elastic.co/downloads/elasticsearch

APM Server download & how to , see: https://www.elastic.co/downloads/apm

Kibana download & how to , see : https://www.elastic.co/downloads/kibana

the dashboard will looks like :
![monitoring](README.assets/monitoring.png)

## faursound CI
TBU



## Deploy in no internet environment

from computer with internet , run below preparation :

```python
mkdir dependencies
pip download -r requirements.txt -d "./dependencies"
tar cvfz dependencies.tar.gz dependencies
```



then go to computer without internet, copy both  dependencies.tar.gz & requirements.txt:

```python
tar zxvf dependencies.tar.gz

pip install --no-index --find-links=dependencies/ -r requirements.txt
```

