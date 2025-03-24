#include "SubGraph.H"

SubGraph::SubGraph()
{
}

SubGraph::~SubGraph()
{
	vertexList.clear();
}

int
SubGraph::addVertex(const char* aName)
{
	string aNameStr(aName);
	vertexList[aNameStr]=0;
	return 0;
}

map<string,int>&
SubGraph::getVertexList()
{
	return vertexList;
}

int
SubGraph::setAttribute(double aVal)
{
	attributes.push_back(aVal);
	return 0;
}


vector<double>& 
SubGraph::getAttributes()
{
	return attributes;
}


bool 
SubGraph::hasSignificantAnnotation(double pVal)
{
	bool isSignificant=false;
	int i=0;
	while ((i<attributes.size()) && (!isSignificant))
	{
		if(attributes[i]<pVal)
		{
			isSignificant=true;
		}
		i++;
	}
	return isSignificant;
}

bool
SubGraph::hasSignificantAnnotation(double pVal,map<int,string>& randomTerms)
{
	bool isSignificant=false;
	int i=0;

	while ((i<attributes.size()) && (!isSignificant))
	{
		if((randomTerms.find(i)==randomTerms.end())&&(attributes[i]<pVal))
		{
			isSignificant=true;
		}
		i++;
	}
	return isSignificant;
}

bool
SubGraph::isEnriched(int attrID,double pVal)
{
	if(attributes[attrID]<pVal)
	{
		return true;
	}
	return false;
}
