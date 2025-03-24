#include <fstream>
#include <iostream>
#include <string.h>
#include <stdlib.h>

#include "GOTerm.H"
#include "GOMgr.H"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

GOMgr::GOMgr()
{
	gotermCnt=0;
}

GOMgr::~GOMgr()
{
}

int 
GOMgr::readGOs(const char* aFName)
{
	ifstream inFile(aFName);
	string strBuffer;
	char* buffer=NULL;
	int buffLength=0;
	while(inFile.good())
	{
		getline(inFile,strBuffer);
		if(strBuffer.length()<=0)
		{
			continue;
		}
		if((buffer==NULL) || (buffLength<=strBuffer.length()))
		{
			if(buffer!=NULL)
			{
				delete[] buffer;
			}
			buffer=new char[strBuffer.length()+1];
		}
		strcpy(buffer,strBuffer.c_str());
		populateGOData(buffer);
	}
	inFile.close();
	cout <<"Read in " << gotermSet.size() << " terms for  "<<  geneSet.size() << " genes" << endl;
	return 0;
}


int
GOMgr::readGOs(py::str fileName)
{
	string aFName = std::string(py::str(fileName));
	ifstream inFile(aFName);
	string strBuffer;
	char* buffer=NULL;
	int buffLength=0;
	while(inFile.good())
	{
		getline(inFile,strBuffer);
		if(strBuffer.length()<=0)
		{
			continue;
		}
		if((buffer==NULL) || (buffLength<=strBuffer.length()))
		{
			if(buffer!=NULL)
			{
				delete[] buffer;
			}
			buffer=new char[strBuffer.length()+1];
		}
		strcpy(buffer,strBuffer.c_str());
		populateGOData(buffer);
	}
	inFile.close();
	cout <<"Read in " << gotermSet.size() << " terms for  "<<  geneSet.size() << " genes" << endl;
	return 0;
}

int
GOMgr::readTermHierarchy(const char* termFName)
{
	ifstream inFile(termFName);
	char aBuffer[1024];
	while(inFile.good())
	{
		inFile.getline(aBuffer,1024);
		if(strlen(aBuffer)<=0)
		{
			continue;
		}
		char* tok=strtok(aBuffer,"\t");
		int tokCnt=0;
		string termName;
		int geneCnt=-1;
		int rootDist=-1;
		while(tok!=NULL)
		{
			if(tokCnt==0)
			{
				termName.append(tok);
			}
			else if(tokCnt==1)
			{
				geneCnt=atoi(tok);
			}
			else if(tokCnt==3)
			{
				rootDist=atoi(tok);
			}
			tok=strtok(NULL,"\t");
			tokCnt++;
		}
		if(rootDist==-1)
		{
			cout <<"No distance from root found " << endl;
			return -1;
		}
		termRootLevel[termName]=rootDist;
	}	
	inFile.close();
	return 0;
}

map<string,GOTerm*>& 
GOMgr::getAllGOs()
{
	return gotermSet;
}

GOTerm* 
GOMgr::getGO(const char* goName)
{
	char termNoDash[1024];
	removeDashes(goName,termNoDash);
	string key(termNoDash);
	if(gotermSet.find(key)==gotermSet.end())
	{
		cout <<"No term with name " << key.c_str() << endl;
		return NULL;
	}
	return gotermSet[key];
}

int
GOMgr::getGOForGene(const char* geneName,map<string,string>& goColorMap, map<string,string>& hitGO)
{
	for(map<string,string>::iterator tIter=goColorMap.begin();tIter!=goColorMap.end();tIter++)
	{
		for(map<string,GOTerm*>::iterator sIter=gotermSet.begin();sIter!=gotermSet.end();sIter++)
		{
			GOTerm* goterm=sIter->second;
			if((strstr(goterm->getGOName(),tIter->first.c_str())==NULL) 
			&& (strstr(tIter->first.c_str(),goterm->getGOName())==NULL))
			{
				continue;
			}
			bool genePresent=goterm->isMember(geneName);
			if(genePresent)
			{
				hitGO[tIter->first]=tIter->second;
			}
		}
	}
	return 0;
}

map<string,string>*
GOMgr::getGOForGene(const char* geneName)
{
	string cgName(geneName);
	if(geneGOSet.find(cgName)==geneGOSet.end())
	{
		return NULL;
	}
	return geneGOSet[cgName];
}

int
GOMgr::getBackgroundDist(map<string,string>& goColorMap, map<string,double>& bgDist)
{
	for(map<string,string>::iterator tIter=goColorMap.begin();tIter!=goColorMap.end();tIter++)
	{
		//Need a map to prevent double counting
		map<string,int> memberGenes;
		for(map<string,GOTerm*>::iterator sIter=gotermSet.begin();sIter!=gotermSet.end();sIter++)
		{
			GOTerm* goterm=sIter->second;
			if((strstr(goterm->getGOName(),tIter->first.c_str())==NULL) 
			&& (strstr(tIter->first.c_str(),goterm->getGOName())==NULL))
			{
				continue;
			}
			map<string,int>& mGenes=goterm->getMemberGenes();
			for(map<string,int>::iterator aIter=mGenes.begin();aIter!=mGenes.end();aIter++)
			{
				memberGenes[aIter->first]=0;
			}
		}
		bgDist[tIter->first]=(double)memberGenes.size()/(double)geneSet.size();
	}
	return 0;
}


int
GOMgr::getBackgroundDist(map<string,double>& bgDist)
{
	int total=0;
	for(map<string,GOTerm*>::iterator sIter=gotermSet.begin();sIter!=gotermSet.end();sIter++)
	{
		GOTerm* goterm=sIter->second;
		int mgeneCnt=goterm->getMemberGenes().size();
		total=total+mgeneCnt;
	}
	for(map<string,GOTerm*>::iterator sIter=gotermSet.begin();sIter!=gotermSet.end();sIter++)
	{
		GOTerm* goterm=sIter->second;
		int mgeneCnt=goterm->getMemberGenes().size();
		bgDist[sIter->first]=(double)mgeneCnt/(double)total;
	}
	return 0;
}

map<string,map<string,string>*>&
GOMgr::getGOForGenes()
{
	return geneGOSet;
}

int
GOMgr::getGeneCnt()
{
	return geneSet.size();
}

//This is a tab-delimited line. The number of tokens per line is the same
int 
GOMgr::populateGOData(char* aBuffer)
{
	char* begin=aBuffer;
	int tokCnt=0;
	string geneName;
	string goName;
	while(begin!=NULL)
	{
		char* end=strchr(begin,'\t');
		if(end!=NULL)
		{
			*end='\0';
		}
		switch(tokCnt)
		{
			case 0:
			{
				geneName.append(begin);
				break;
			}
			case 1:
			{
				goName.append(begin);
				break;
			}
			default:
			{
				break;
			}
		}
		tokCnt++;
		if(end!=NULL)
		{
			begin=end+1;
		}
		else
		{
			begin=NULL;
		}
	}
	if(geneName.length()==0)
	{
		return 0;
	}
	GOTerm* goterm=NULL;
	if(gotermSet.find(goName)==gotermSet.end())
	{
		goterm=new GOTerm;
		goterm->setGOName(goName.c_str());
		gotermSet[goName]=goterm;
	}
	else
	{
		goterm=gotermSet[goName];
	}
	goterm->addMemberGene(geneName.c_str());
	geneSet[geneName]=0;
	map<string,string>* geneGOInfo=NULL;
	if(geneGOSet.find(geneName)==geneGOSet.end())
	{
		geneGOInfo=new map<string,string>;
		geneGOSet[geneName]=geneGOInfo;
	}
	else
	{
		geneGOInfo=geneGOSet[geneName];
	}
	return 0;
}

int 
GOMgr::getGeneCntForTerm(const char* termName) 
{
	char termNoDash[1024];
	removeDashes(termName,termNoDash);
	string termKey(termNoDash);
	GOTerm* term=getGO(termNoDash);
	if(term==NULL)
	{
		cout <<"No term by name " << termKey.c_str() << endl;
		return 0;
	}
	return term->getMemberCnt();
}

int 
GOMgr::getRootDistForTerm(const char* termName)
{
	int level=0;
	char termNoDash[1024];
	removeDashes(termName,termNoDash);
	string termKey(termNoDash);
	map<string,int>::iterator tIter=termRootLevel.find(termKey);
	if(tIter==termRootLevel.end())
	{
		cout <<"No term with name " << termKey.c_str() << endl;
		return 0;
	}
	return tIter->second;
}

int
GOMgr::removeDashes(const char* termName, char* termNoDash)
{
	int i=0;
	while(termName[i]!='\0')
	{
		if(termName[i]=='_')
		{
			//termNoDash[i]=' ';
			termNoDash[i]=termName[i];
		}
		else
		{
			termNoDash[i]=termName[i];
		}
		i++;
	}
	termNoDash[i]='\0';
	return 0;
}
