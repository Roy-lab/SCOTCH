#include "GOTerm.H"

GOTerm::GOTerm()
{
}

GOTerm::~GOTerm()
{
}

//Although we are saying tissue, really we mean tissue and stage development
int 
GOTerm::setGOName(const char* aName)
{
	goTerm.append(aName);
	return 0;
}

const char*
GOTerm::getGOName()
{
	return goTerm.c_str();
}

int 
GOTerm::addMemberGene(const char* aName)
{
	string geneName(aName);
	memberGenes[geneName]=0;
	return 0;
}

map<string,int>& 
GOTerm::getMemberGenes()
{
	return memberGenes;
}

int
GOTerm::getMemberCnt()
{
	return memberGenes.size();
}

bool
GOTerm::isMember(const string& geneName)
{
	if(memberGenes.find(geneName)==memberGenes.end())
	{
		return false;
	}
	return true;
}


bool
GOTerm::isMember(const char* geneName)
{
	string key(geneName);
	if(memberGenes.find(key)==memberGenes.end())
	{
		return false;
	}
	return true;
}
