#include <string>
#include <string.h>
#include <iostream>
#include <fstream>

#include <math.h>
#include <stdlib.h>


#include "Randomizer.H"
#include "SubGraph.H"
#include "SubGraphMgr.H"

using namespace std;


SubGraphMgr::SubGraphMgr()
{
}

SubGraphMgr::~SubGraphMgr()
{
}

int 
SubGraphMgr::readSubGraphs(const char* aFName)
{
	ifstream inFile(aFName);
	char* buffer=NULL;
	int buffSize=0;
	string strBuff;
	int sgid=0;
	while(inFile.good())
	{
		getline(inFile,strBuff);
		if(strBuff.length()<=0)
		{
			continue;
		}
		if(buffSize<=strBuff.length())
		{
			if(buffer!=NULL)
			{
				delete[] buffer;
			}
			buffSize=strBuff.length()+1;
			buffer=new char[buffSize];
		}
		strcpy(buffer,strBuff.c_str());
		char* tok=strtok(buffer,"\t#");
		SubGraph* sg=new SubGraph;
		int tokCnt=0;
		string sgName;
		while(tok!=NULL)
		{
			if(tokCnt==0)
			{
				sgName.append(tok);
			}
			else
			{
				sg->addVertex(tok);
			}
			tokCnt++;
			tok=strtok(NULL,"\t#");
		}
		subGraphSet.push_back(sg);
		idNameMap[sgid]=sgName;
		sgid++;
	}
	inFile.close();
	return 0;
}


int
SubGraphMgr::initSubGraphsFromPython(const py::dict& assignmentDict, const py::str& cluster_col)
{
	std::string geneID;
	std::string sgName;
	py::dict varData;
	int sgid(0);

	for(const auto& subGraphData : assignmentDict)
	{
		geneID = std::string(py::str(subGraphData.first));
		varData = py::cast<py::dict>(subGraphData.second);
    		sgName = std::string(py::str(varData[cluster_col]));

    		if (subGraphIdMap.find(sgName) == subGraphIdMap.end())
    		{
    			SubGraph* sg = new SubGraph;
    			sg->addVertex(geneID.c_str());
    			subGraphSet.push_back(sg);
    			idNameMap[sgid]=sgName;
    			subGraphIdMap[sgName] = sgid;
    			sgid++;
    		}
		else
		{
			SubGraph* sg = subGraphSet[subGraphIdMap[sgName]];
			sg->addVertex(geneID.c_str());
		}
	}
  return 0;
}


int
SubGraphMgr::readSubGraphAttributes(const char* attrFName)
{
	ifstream inFile(attrFName);
	string strBuff;
	char* buffer=NULL;
	int buffLen=0;
	int lineCnt=0;
	while(inFile.good())
	{
		getline(inFile,strBuff);
		if(strBuff.length()<=0)
		{
			continue;
		}
		if(buffLen<=strBuff.length())
		{
			buffLen=strBuff.length()+1;
			if(buffer!=NULL)
			{
				delete[] buffer;
			}
			buffer=new char[buffLen];
		}
		strcpy(buffer,strBuff.c_str());
		if(lineCnt==0)
		{
			char* tok=strtok(buffer,"\t");
			int tokCnt=0;
			while(tok!=NULL)
			{
				if(tokCnt>0)
				{
					string attrName(tok);
					attributeNames[attrName]=tokCnt-1;
				}
				tok=strtok(NULL,"\t");
				tokCnt++;
			}
		}
		else
		{
			SubGraph* sg=subGraphSet[lineCnt-1];
			char* tok=strtok(buffer,"\t");
			int tokCnt=0;
			while(tok!=NULL)
			{
				if(tokCnt>0)
				{
					double attrVal=atof(tok);
					double logpval=0;
					if(attrVal>0)
					{
						logpval=log(attrVal);
					}
					else
					{
						cout <<"Found 0 pval for SG" << lineCnt-1 << endl;
						logpval=1;
					}
					//sg->setAttribute(logpval);
					sg->setAttribute(attrVal);
				}
				tok=strtok(NULL,"\t");
				tokCnt++;
			}
		}
		lineCnt++;
	}
	cout << " Read " << attributeNames.size() << " attributes for " << subGraphSet.size() << " different graphs " << endl; 
	inFile.close();
	return 0;
}

int
SubGraphMgr::readGeneList(const char* aFName)
{
	ifstream inFile(aFName);
	char buffer[1024];
	int geneId=0;
	while(inFile.good())
	{
		inFile.getline(buffer,1023);
		if(strlen(buffer)<=0)
		{
			continue;
		}
		char* pos=strchr(buffer,'\t');
		if(pos!=NULL)
		{
			*pos='\0';	
		}
		string geneName(buffer);
		geneNameIDMap[geneName]=geneId;
		geneIDNameMap[geneId]=geneName;
		geneId++;
	}
	inFile.close();
	return 0;
}

int SubGraphMgr::initGeneListFromPython(const py::list& gene_list)
{
	string gene_str;
	int geneId(0);

	for (const auto& gene : gene_list)
	{
		gene_str = std::string(py::str(gene));
		geneNameIDMap[gene_str] = geneId;
		geneIDNameMap[geneId] = gene_str;
		geneId++;
	}
	return 0;
}


int
SubGraphMgr::genRandomGraphs(int totalIterations,int sgSize)
{
	Randomizer rand;
	rand.initialize(0,geneNameIDMap.size()-1);
	for(int i=0;i<subGraphSet.size();i++)
	{
		SubGraph* sg=subGraphSet[i];
		delete sg;
	}
	subGraphSet.clear();
	for(int i=0;i<totalIterations;i++)
	{
		SubGraph* sg=new SubGraph;
		while(sg->getVertexList().size()<sgSize)
		{
			int nodeId=rand.getRandomNumber();
			if(geneIDNameMap.find(nodeId)==geneIDNameMap.end())
			{
				cout << "Bad node id " << nodeId << endl;
				return -1;
			}
			sg->addVertex(geneIDNameMap[nodeId].c_str());
		}
		subGraphSet.push_back(sg);
	}

	return 0;
}


int
SubGraphMgr::genRandomGraphs(int totalIterations,int sgSize, vector<SubGraph*>& randsgSet)
{
	Randomizer rand;
	rand.initialize(0,geneNameIDMap.size()-1);
	for(int i=0;i<totalIterations;i++)
	{
		SubGraph* sg=new SubGraph;
		while(sg->getVertexList().size()<sgSize)
		{
			int nodeId=rand.getRandomNumber();
			if(geneIDNameMap.find(nodeId)==geneIDNameMap.end())
			{
				cout << "Bad node id " << nodeId << endl;
				return -1;
			}
			sg->addVertex(geneIDNameMap[nodeId].c_str());
		}
		randsgSet.push_back(sg);
	}

	return 0;
}


vector<SubGraph*>& 
SubGraphMgr::getSubGraphs()
{
	return subGraphSet;
}

map<string,int>&
SubGraphMgr::getAttributeNames()
{
	return attributeNames;
}

SubGraph*
SubGraphMgr::getGraphAt(int i)
{
	return subGraphSet[i];
}

int 
SubGraphMgr::getGeneCnt()
{
	return geneNameIDMap.size();
}

map<string,int>&
SubGraphMgr::getGeneList()
{
	return geneNameIDMap;
}


string& 
SubGraphMgr::getSGName(int sgid)
{
	if(idNameMap.find(sgid)==idNameMap.end())
	{
		cout <<"No SGID " << sgid << endl;
		exit(0);
	}
	return idNameMap[sgid];
}
