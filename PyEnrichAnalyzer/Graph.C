#include <fstream>
#include <iostream>
#include <queue>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <sys/timeb.h>
#include <sys/time.h>
#include <time.h>

#include "Distance.H"
#include "Vertex.H"
#include "Graph.H"
#include "Clique.H"


Graph::Graph()
{
}

Graph::~Graph()
{
}

//The format of the file is with an edge information per line
int 
Graph::makeGraph(const char* fName)
{
	ifstream inFile(fName);
	char buffer[1024];
	while(inFile.good())
	{
		inFile.getline(buffer,1023);
		if(strlen(buffer)==0)
		{
			continue;
		}
		char* tok=strtok(buffer,"\t");
		int tokCnt=0;
		Vertex* v=NULL;
		Vertex* u=NULL;
		while(tok!=NULL)
		{
			if(tokCnt==0)
			{
				string vKey(tok);
				if(vertexList.find(vKey)==vertexList.end())
				{
					v=new Vertex;
					v->setName(tok);
					vertexList[vKey]=v;
				}
				else
				{
					v=vertexList[vKey];
				}
			}
			else if(tokCnt==1)
			{
				string uKey(tok);
				if(vertexList.find(uKey)==vertexList.end())
				{
					u=new Vertex;
					u->setName(tok);
					vertexList[uKey]=u;
				}
				else
				{
					u=vertexList[uKey];
				}
			}
			tokCnt++;
			tok=strtok(NULL,"\t");
		}
		if(v!=NULL && u!=NULL)
		{
			v->setOutNeighbour(u);
			u->setOutNeighbour(v);
			//u->setInNeighbour(v);
		}
	}
	inFile.close();
	cout <<"Number of nodes: "<< vertexList.size() <<   endl;
	return 0;
}

int
Graph::dumpGraph()
{
	ofstream oFile("graph.sif");
	map<string,int> doneVertices;
	for(map<string,Vertex*>::iterator aIter=vertexList.begin();aIter!=vertexList.end();aIter++)
	{
		NINFO_MAP& neighbrs=aIter->second->getImmediateNeighbours();
		for(NINFO_MAP_ITER nmIter=neighbrs.begin();nmIter!=neighbrs.end();nmIter++)
		{
			if(doneVertices.find(nmIter->first)==doneVertices.end())
			{
//				oFile << aIter->first.c_str() <<" interacts " << nmIter->first.c_str() << endl;
				oFile << aIter->first.c_str() <<"\t" << nmIter->first.c_str() << endl;
				oFile << nmIter->first.c_str() <<"\t" << aIter->first.c_str() << endl;
			}
		}
		doneVertices[aIter->first]=0;
	}
	oFile.close();
	return 0;
}

int
Graph::showNodeProperties(const char* nodeFName)
{
	ofstream oFile(nodeFName);
	oFile << "NodeType\tNodeName\tOutDegree\tInDegree\tReachableNodes\tCloseness"
		<<"\tBridgeness"
		<<"\tCoregulators\tAvgCommTargets"
		//<<"\tSdCommTargets"
		<<"\tAvgPathLength"
		//<<"\tSdPathLength"
		<<"\tAvgTargetIndegree"
		<<"\tAvgTargetReachableIndegree"
		<< endl;
	for(map<string,Vertex*>::iterator aIter=vertexList.begin();aIter!=vertexList.end();aIter++)
	{
		Vertex* v=aIter->second;
		if(v->getOutDegree()==0)
		{
			if((v->getCloseness()>0) || (v->getBetweenness()>0))
			{
	//			cout <<"Found a non-tf node node with closeness " << v->getCloseness()
	//				<<" and betweenness " << v->getBetweenness() << endl;
					
			}
			continue;
		}
		oFile << "TF\t"<< aIter->first.c_str() << "\t" << v->getOutDegree() 
			<< "\t" << v->getInDegree()
			<<"\t" << v->getReachableOutDegree()
			<< "\t" << v->getCloseness()
			//<<"\t" << v->getBetweenness()
			//<<"\t" << v->getBetweennessNorm()
			<<"\t" << v->getBridgeness()
			<<"\t" << v->getCoRegulators()
			<<"\t" << v->getAvgCommonTargets()
			//<<"\t" << v->getSdCommonTargets()
			<<"\t" << v->getAvgPathLength()
			//<<"\t" << v->getSdPathLength()
			<<"\t" << v->getAvgTargetIndegree()
			<<"\t" << v->getAvgTargetReachableIndegree()
			<<endl;
	}
	oFile <<"NodeType\tNodeName\tOutDegree\tInDegree\tRegPathRatio\tRegulatibity\tTFOutDegree\tTFReachableOutDegree" << endl;
	for(map<string,Vertex*>::iterator aIter=vertexList.begin();aIter!=vertexList.end();aIter++)
	{
		Vertex* aTarget=aIter->second;
		if(aTarget->getInDegree()==0)
		{
			continue;
		}
		oFile <<"Target\t" << aIter->first.c_str() 
			<<"\t"<<aTarget->getOutDegree()
			<< "\t"<< aTarget->getInDegree() 
			<<"\t" << aTarget->getRegPathRatio() 
			<< "\t" << aTarget->getRegulatibility()
			<<"\t" << aTarget->getAvgTFOutdegree()
			<<"\t" << aTarget->getAvgTFReachableOutdegree()
			<< endl;
	}
	oFile.close();
	return 0;
}

//This is a bad way of doing stuff. 
int 
Graph::obtainConnectivity()
{
	map<string,Vertex*>::iterator aIter;
	/*struct timeb begintime;
	struct timeb endtime;*/
	struct timeval begintime;
	struct timeval endtime;
	gettimeofday(&begintime,NULL);
	for(aIter=vertexList.begin();aIter!=vertexList.end();aIter++)
	{
		Vertex* v=aIter->second;
		v->findReachableNodes();
	}
	gettimeofday(&endtime,NULL);

	cout << "Time elapsed" << endtime.tv_sec-begintime.tv_sec<< " seconds and " << endtime.tv_usec-begintime.tv_usec << " micro secs" << endl;

	//computePathLenFrequency();
	//computeAveragePathLength();
	return 0;
}

int
Graph::getShortestPathLen(const char* vName,const char* uName)
{
	string vKey(vName);
	if(vertexList.find(vKey)==vertexList.end())
	{
		cout <<"No vertex by name of " << vName << endl;
		return 0;
	}
	Vertex* v=vertexList[vKey];
	string uKey(uName);
	Vertex* u=vertexList[uKey];
	int pLen=v->getPathLength(u);
	if(pLen==-1)
	{
	//	pLen=vertexList.size();
	}
	return pLen;
}

int 
Graph::showConnectivity()
{
	for(map<string,Vertex*>::iterator aIter=vertexList.begin(); aIter!=vertexList.end();aIter++)
	{
		Vertex* v=aIter->second;
		v->showReachability(cout);
	}
	return 0;
}

int
Graph::showPathLenFrequency(const char* aFName)
{
	ofstream oFile(aFName);
	oFile <<"Pathlen\tNumber" << endl;
	for(map<int,int>::iterator aIter=pathLenFreqCnt.begin();aIter!=pathLenFreqCnt.end();aIter++)
	{
		oFile << aIter->first << "\t" << aIter->second << endl;
	}
	
	
	oFile <<"Harmonic mean of l " << harmonicMeanLength << endl;
	oFile <<"Assortativity Out-In " << outIncorrCoeff << endl;
	oFile <<"Reachable Assortativity Out-In " << reachoutIncorrCoeff << endl;
	oFile <<"Assortativity Out-Out " << outOutcorrCoeff << endl;
	oFile <<"Total paths " << totalPaths << endl;
	oFile <<"Average path length " << avgPathLength << endl;
	oFile <<"Stdev path length " << sdPathLength << endl;
	oFile <<"Average regpath ratio " << avgRegPathRatio << endl;
	oFile <<"Stdev regpath ratio " << sdRegPathRatio << endl;
	oFile <<"Average regulatibility " << avgRegulatibility << endl;
	oFile <<"Stdev regulatibility " << sdRegulatibility << endl;
	oFile <<"Average betweenness "<< avgBetweenness<< endl;
	oFile <<"Average bridgeness "<< avgBridgeness<< endl;
	oFile <<"Average reachability diff "<< avgPureReachabilityDiff<< endl;
	oFile.close();
	return 0;
}

//A slightly better way of getting the all pairs shortest path
int
Graph::obtainConnectivity_Efficient()
{
	struct timeval begintime;
	struct timeval endtime;
	//First push every vertex on the untouched vertex map
	for(map<string,Vertex*>::iterator aIter=vertexList.begin();aIter!=vertexList.end();aIter++)
	{
		untouchedVertexMap[aIter->first]=aIter->second;
		aIter->second->setQueueStatus(Vertex::NO_QUEUE);
		aIter->second->setGraph(this);
	}
	gettimeofday(&begintime,NULL);
	
	while(!untouchedVertexMap.empty())
	{
		//Start from any vertex, lets say the first one
		map<string,Vertex*>::iterator aIter=untouchedVertexMap.begin();
		Vertex* start=aIter->second;
		untouchedVertexMap.erase(aIter);
		queue<Vertex*> globalQueue;
		globalQueue.push(start);
		start->setQueueStatus(Vertex::ON_QUEUE);
		while(!globalQueue.empty())
		{
			Vertex* u=globalQueue.front();
			globalQueue.pop();
			//Push all the immediate neighbours that have not been pushed yet, on to the queue
			NINFO_MAP& vNeigh=u->getImmediateNeighbours();
			for(NINFO_MAP_ITER nmIter=vNeigh.begin();nmIter!=vNeigh.end();nmIter++)
			{
				Vertex* v=nmIter->second;
				if(v->getQueueStatus()==Vertex::NO_QUEUE)
				{
					globalQueue.push(v);
					v->setQueueStatus(Vertex::ON_QUEUE);
					map<string,Vertex*>::iterator aIter=untouchedVertexMap.find(nmIter->first);
					if(aIter==untouchedVertexMap.end())
					{
						cout <<"There is a problem " << endl;
						exit(0);
					}
					untouchedVertexMap.erase(aIter);
				}
			}
			//Now for all neighbours of u that have not been popped off from the queue, add as u as their reachable neighbours
			for(NINFO_MAP_ITER nmIter=vNeigh.begin();nmIter!=vNeigh.end();nmIter++)
			{
				Vertex* v=nmIter->second;
				if(v->getQueueStatus()!=Vertex::OFF_QUEUE)
				{
					if(v->getReachableOutDegree()==0)
					{
						v->setReachableNeighbour(u,1);
						u->addReachableNeighbour(v,1);
					}
					else
					{
						v->addReachableNeighbour(u,1);
						u->addReachableNeighbour(v,1);
					}
				}
			}
			u->setQueueStatus(Vertex::OFF_QUEUE);
		}
	}

	gettimeofday(&endtime,NULL);
	cout << "Time elapsed" << endtime.tv_sec-begintime.tv_sec<< " seconds and " << endtime.tv_usec-begintime.tv_usec << " micro secs" << endl;

	
	return 0;
}

//For a graph to be connected every vertex must be reachable to every other vertex
bool 
Graph::isConnected()
{
	bool isReachable=true;
	map<string,Vertex*>::iterator aIter=vertexList.begin();
	while(aIter!=vertexList.end() && isReachable)
	{
		Vertex* u=aIter->second;
		map<string,Vertex*>::iterator bIter=vertexList.begin();
		while(bIter!=vertexList.end() && isReachable)
		{
			if(aIter!=bIter)
			{
				Vertex* v=bIter->second;
				if(!u->isReachable(v))
				{
					isReachable=false;
				}
			}
			bIter++;
		}
		aIter++;
	}
	return isReachable;
}

double
Graph::computeAveragePathLength()
{
	totalPaths=0;
	int totalPathLength=0;
	for(map<string,Vertex*>::iterator aIter=vertexList.begin();aIter!=vertexList.end();aIter++)
	{
		Vertex* u=aIter->second;
		NDIST_MAP& reachableNeighbours=u->getReachableNeighbours();
		for(NDIST_MAP_ITER nmIter=reachableNeighbours.begin();nmIter!=reachableNeighbours.end();nmIter++)
		{
			int pLen=nmIter->second;
			totalPaths++;
			totalPathLength=totalPathLength+pLen;
		}
	}
	avgPathLength=(double)totalPathLength/(double)totalPaths;
	sdPathLength=0;
	for(map<string,Vertex*>::iterator aIter=vertexList.begin();aIter!=vertexList.end();aIter++)
	{
		Vertex* u=aIter->second;
		NDIST_MAP& reachableNeighbours=u->getReachableNeighbours();
		for(NDIST_MAP_ITER nmIter=reachableNeighbours.begin();nmIter!=reachableNeighbours.end();nmIter++)
		{
			int pLen=nmIter->second;
			double diff=(double) pLen-avgPathLength;
			sdPathLength=sdPathLength+(diff*diff);
		}
	}
	sdPathLength=sqrt(sdPathLength/((double)(totalPaths-1)));
	return avgPathLength;
}

int
Graph::computeAverageBetweenness()
{
	double nodeCnt=0;
	avgBetweenness=0;
	avgBetweennessNorm=0;
	avgBridgeness=0;
	for(map<string,Vertex*>::iterator aIter=vertexList.begin();aIter!=vertexList.end();aIter++)
	{
		Vertex* v=aIter->second;
		if(v->getBetweenness()>0)
		{
			avgBetweenness=avgBetweenness+v->getBetweenness();
			avgBetweennessNorm=avgBetweennessNorm+v->getBetweennessNorm();
			avgBridgeness=avgBridgeness+ v->getBridgeness();
			nodeCnt++;
		}
	}
	avgBetweenness=avgBetweenness/nodeCnt;
	avgBetweennessNorm=avgBetweennessNorm/nodeCnt;
	avgBridgeness=avgBridgeness/nodeCnt;
	return 0;
}

int
Graph::computeAveragePureReachabilityDiff()
{
	double tfWithReg=0;
	double tfWithoutReg=0;
	double avgPureTarget1=0;
	double avgPureTarget2=0;
	for(map<string,Vertex*>::iterator aIter=vertexList.begin();aIter!=vertexList.end();aIter++)
	{
		Vertex* v=aIter->second;
		if(v->getOutDegree()==0)
		{
			continue;
		}
		if(v->getInDegree()==0)
		{
			avgPureTarget1=avgPureTarget1+(double)v->getPureReachableNodes();
			tfWithoutReg++;
		}
		else 
		{
			avgPureTarget2=avgPureTarget2+(double)v->getPureReachableNodes();
			tfWithReg++;
		}
	}
	avgPureReachabilityDiff=(avgPureTarget2/tfWithReg)-(avgPureTarget1/tfWithoutReg);
	return 0;
}
			
int
Graph::getTotalPathCnt()
{
	return totalPaths;
}

int
Graph::getPathLength(const char* uName, const char* vName, int& pathLength)
{
	string uKey(uName);
	string vKey(vName);
	if((vertexList.find(uKey)==vertexList.end())|| (vertexList.find(vKey)==vertexList.end()))
	{
		//vertex does not exist in this graph
		pathLength=-2;
	}
	else
	{
		Vertex* u=vertexList[uKey];
		Vertex* v=vertexList[vKey];
	
		pathLength=u->getPathLength(v);
	}
	return 0;
}

//We could use several measures, e.g. KL divergence between the path length distribution of the two graphs
//We could also use the absolute difference in path lengths between the same pair of vertices
//Here if a path exists in one graph and not in another we keep it in different variable
int 
Graph::computeGraphDistance(Graph* g)
{
	double pathNotFound=0;
	//Total number of paths of len 1 in the original graph that were not found in the predicted graph
	double pathsNotFound_l1=0;
	//Average path len of paths of len 1 in the original graph that were found in the predicted graph
	double pathLenFound_l1=0;
	double pathDiff=0;
	double correctPathLen=0;
	double wrongPathLen=0;

	double avg1=computeAveragePathLength();
	double avg2=g->computeAveragePathLength();
	cout <<"Difference in path length average " << fabs(avg1-avg2) << endl; 
	//Stores the number of paths of different lengths. Key is the path length in the original graph
	//Value is the number of paths which are of different length
	map<int,double> pathDiffSum;
	//Store the difference in path lengths. Key is the path length in the original graph
	//Value is the sum of the path differences of this particular length specified by the key value
	map<int,int> pathDiffCnt;

	//Store the difference distribution between path lengths. Key is the path length in the original graph
	//Value is the distribution of the path differences
	map<int,map<int,int>*> pathDiffDist;
	
	map<string,Vertex*>::iterator aIter;
	for(aIter=vertexList.begin();aIter!=vertexList.end(); aIter++)
	{
		Vertex* u=aIter->second;
		map<string,Vertex*>::iterator bIter;
		for(bIter=vertexList.begin();bIter!=vertexList.end();bIter++)
		{
			if(aIter!=bIter)
			{
				Vertex* v=bIter->second;
				int pathLen1=0;
				getPathLength(u->getName(),v->getName(),pathLen1);
				if(pathLen1<0)
				{
					continue;
				}

				int pathLen2=0;
				g->getPathLength(u->getName(),v->getName(),pathLen2);
				int diff=abs(pathLen1-pathLen2);
				if(pathLen2==-2 || pathLen2==-1)
				{
					pathNotFound++;
					if(pathLen1==1)
					{
						pathsNotFound_l1++;
					}
				}
				else
				{
					if(pathLen1==1)
					{
						pathLenFound_l1+=pathLen2;
					}
					if(diff==0)
					{
						correctPathLen++;
					}
					else
					{
						wrongPathLen++;
					}
				}
				pathDiff=pathDiff+abs(diff);
				if(pathDiffSum.find(pathLen1)==pathDiffSum.end())
				{
					pathDiffSum[pathLen1]=(double)diff/(double)pathLen1;
					pathDiffCnt[pathLen1]=1;
				}
				else
				{
					pathDiffCnt[pathLen1]=pathDiffCnt[pathLen1]+1;
					pathDiffSum[pathLen1]=pathDiffSum[pathLen1]+(double)diff/(double)pathLen1;
				}	
				if(pathDiffDist.find(pathLen1)==pathDiffDist.end())
				{
					map<int,int>* iVect=new map<int,int>;
					(*iVect)[diff]=1;
					pathDiffDist[pathLen1]=iVect;
				}
				else
				{
					map<int,int>* iVect=pathDiffDist[pathLen1];
					if(iVect->find(diff)==iVect->end())
					{
						(*iVect)[diff]=1;
					}
					else
					{
						(*iVect)[diff]=(*iVect)[diff]+1;
					}
				}
				
			}
		}
	}
	double  pathFound=correctPathLen + wrongPathLen;
	cout << "Percentage paths found " << pathFound/(double)totalPaths << endl;
	cout << "Path length recall " << correctPathLen/(double)totalPaths<< endl;
	cout << "Path length precision " << correctPathLen/(double)g->getTotalPathCnt()<< endl;

	cout << "Average path difference " << pathDiff/(double) totalPaths <<  endl;
	
	//Normalized allows us to add importance to paths of shorter lengths in the original graph
	double avgNormPathDiff=0;
	for(map<int,double>::iterator aIter=pathDiffSum.begin();aIter!=pathDiffSum.end();aIter++)
	{
		double pathDiff=aIter->second;
		int diffCnt=pathDiffCnt[aIter->first];
		double avgPDforlen=pathDiff/(double)diffCnt;
		avgNormPathDiff=avgNormPathDiff+avgPDforlen;
	}
	cout << "Average normalized path difference "<< avgNormPathDiff <<endl;

	/*cout <<"Difference in path lengths for all paths of a specific length in the original graph" << endl;
	for(map<int,map<int,int>*>::iterator aIter=pathDiffDist.begin();aIter!=pathDiffDist.end();aIter++)
	{
		map<int,int>* iMap=aIter->second;
		cout <<"Original path len " << aIter->first << endl;
		cout <<"Diff\tCount "<< endl;
		for(map<int,int>::iterator bIter=iMap->begin();bIter!=iMap->end();bIter++)
		{
			cout << bIter->first << "\t" << bIter->second << endl;
		}
	}*/
	double pathsOfl1=(double)pathLenFreqCnt[1];
	cout << "Percentage pairs not found " <<pathsNotFound_l1/pathsOfl1<< endl;
	cout << "Average distance between pairs in predicted graph " <<pathLenFound_l1/(pathsOfl1-pathsNotFound_l1) << endl;
	return 0;
}

int
Graph::computeSymmetricKLDivergence(Graph* g)
{
	smoothPathLenDist(g);
	g->smoothPathLenDist(this);
	double kl1=getKLDivergence(g);
	double kl2=g->getKLDivergence(this);
	cout <<"KL Divergence " << kl1 + kl2<< endl;
	return 0;
}

double
Graph::getKLDivergence(Graph* g)
{
	double klDiv=0;
	for(map<int,double>::iterator aIter=pathLenDist.begin();aIter!=pathLenDist.end();aIter++)
	{
		double p1=aIter->second;
		if(g->pathLenDist.find(aIter->first)==g->pathLenDist.end())
		{
			cout <<"Second graph does not have any prob for len=" << aIter->first << endl;
			continue;
		}
		else
		{
			double p2=g->pathLenDist[aIter->first];
			double temp1=p1*(log(p1/p2));
			klDiv=klDiv+temp1;	
		}
	}
	return klDiv;
}


int
Graph::computePathLenFrequency()
{
	double totalPaths=0;
	map<string,Vertex*>::iterator aIter;
	for(aIter=vertexList.begin();aIter!=vertexList.end(); aIter++)
	{
		Vertex* u=aIter->second;
		map<string,Vertex*>::iterator bIter;
		for(bIter=vertexList.begin();bIter!=vertexList.end();bIter++)
		{
			if(aIter!=bIter)
			{
				Vertex* v=bIter->second;
				int pathLen1=0;
				getPathLength(u->getName(),v->getName(),pathLen1);
				if(pathLen1>0)
				{
					if(pathLenFreqCnt.find(pathLen1)==pathLenFreqCnt.end())
					{
						pathLenFreqCnt[pathLen1]=1;
						pathLenDist[pathLen1]=1.0;
					}
					else
					{
						pathLenFreqCnt[pathLen1]=pathLenFreqCnt[pathLen1]+1;
						pathLenDist[pathLen1]=pathLenDist[pathLen1]+1.0;
					}
					harmonicMeanLength=harmonicMeanLength+(1.0/(double)pathLen1);
				}
				totalPaths++;
			}
		}
	}
	harmonicMeanLength=harmonicMeanLength/totalPaths;
	return 0;
}


int 
Graph::smoothPathLenDist(Graph* g)
{
	//Smooth the distributions
	//Smooth this distribution for the path length values in the graph g
	for(map<int,double>::iterator aIter=g->pathLenDist.begin();aIter!=g->pathLenDist.end();aIter++)
	{
		if(pathLenDist.find(aIter->first)==pathLenDist.end())
		{
			pathLenDist[aIter->first]=0;
		}
	}
	double totalPath=0;
	//Add 1 for smoothing
	for(map<int,double>::iterator aIter=pathLenDist.begin();aIter!=pathLenDist.end();aIter++)
	{
		aIter->second=aIter->second+1;
		totalPath=totalPath+aIter->second;
	}
	//Now normalize to get a probability distribution
	for(map<int,double>::iterator aIter=pathLenDist.begin();aIter!=pathLenDist.end();aIter++)
	{
		aIter->second=aIter->second/totalPath;
	}
	return 0;
}


int 
Graph::getDegreeDist()
{
	//true for indegree
	getDegreeDist(true,indegreeDist);
	getDegreeDist(false,outdegreeDist);
	return 0;
}

int 
Graph::showDegreeDist(const char* fName)
{
	ofstream oFile(fName);
	oFile <<"InDegree\tNodeCnt"<< endl;
	for(map<int,int>::iterator aIter=indegreeDist.begin();aIter!=indegreeDist.end();aIter++)
	{
		oFile <<aIter->first <<"\t" << aIter->second << endl;
	}
	oFile <<"OutDegree\tNodeCnt" << endl;
	for(map<int,int>::iterator aIter=outdegreeDist.begin();aIter!=outdegreeDist.end();aIter++)
	{
		oFile <<aIter->first <<"\t" << aIter->second << endl;
	}
	
	oFile.close();
	return 0;
}

int 
Graph::getNodesWithDegreeAtleast(int degree, const char* fName, const char* hName)
{
	vector<int> degVect;
	vector<string> nameVect;
	for(map<string,Vertex*>::iterator aIter=vertexList.begin(); aIter!=vertexList.end();aIter++)
	{
		Vertex* v=aIter->second;
		degVect.push_back(v->getOutDegree());
		nameVect.push_back(aIter->first);
	}
	//Sort by degree
	for(int i=0;i<degVect.size();i++)
	{
		for(int j=i+1;j<degVect.size();j++)
		{
			if(degVect[i]<degVect[j])
			{
				int tempDeg=degVect[i];
				degVect[i]=degVect[j];
				degVect[j]=tempDeg;
				string tempStr(nameVect[i].c_str());
				nameVect[i].erase();
				nameVect[i].append(nameVect[j].c_str());
				nameVect[j].erase();
				nameVect[j].append(tempStr.c_str());
			}
		}
	}
	int highDegreeNodes=0;
	map<string,int> doneVertices;
	ofstream oFile(fName);
	ofstream hFile(hName);
	for(int i=0;i<degVect.size();i++)
	{
		if(degVect[i]>=degree)
		{
			//oFile <<nameVect[i].c_str() << "\t" << degVect[i] << endl;
			Vertex* v=vertexList[nameVect[i]];
			NINFO_MAP& neighbrSet=v->getImmediateNeighbours();
			for(NINFO_MAP_ITER nIter=neighbrSet.begin();nIter!=neighbrSet.end();nIter++)
			{
				if(vertexList[nIter->first]->getOutDegree()<degree)
				{
			//		continue;
				}
				if(doneVertices.find(nIter->first)==doneVertices.end())
				{
					oFile << nameVect[i].c_str()<<"\t"<< nIter->first.c_str() << endl;
				//	oFile << nIter->first.c_str()<<"\t"<< nameVect[i].c_str() << endl;
				}
			}
			hFile << nameVect[i].c_str() << endl;
			highDegreeNodes++;
			//doneVertices[nameVect[i]]=0;
		}
	}
	oFile.close();
	hFile.close();
	cout <<"Found " << highDegreeNodes << " of degree >= " << degree << endl;
	return 0;
}

Vertex*
Graph::getVertex(const char* vname)
{
	Vertex* v=NULL;
	string key(vname);
	map<string,Vertex*>::iterator aIter=vertexList.find(key);
	if(aIter==vertexList.end())
	{
		cout <<"Warning! Null pointer returned for " << vname << endl;
	}
	else
	{
		v=aIter->second;
	}
	return v;
}


int 
Graph::findCliques()
{
	for(map<string,Vertex*>::iterator aIter=vertexList.begin();aIter!=vertexList.end();aIter++)
	{
		Vertex* v=aIter->second;
		for(int i=0;i<cliqueSet.size();i++)
		{
			Clique* clq=cliqueSet[i];
			if(clq->inClique(v))
			{
				clq->addMember(v);
			}
		}
		Clique* newClq=new Clique;
		newClq->addMember(v);
		cliqueSet.push_back(newClq);
	}
	return 0;
}

int 
Graph::showCliques(const char* aFName,int minClq)
{
	ofstream oFile(aFName);
	for(int i=0;i<cliqueSet.size();i++)
	{
		Clique* clq=cliqueSet[i];
		clq->showClique(oFile,minClq);
	}
	oFile.close();
	return 0;
}


//Here we simply want to compute the correlation coefficient between two vectors, where
//the number of elements of the vector correspond to the number directed edges in the network
//In one vector we put the out degree of the source node and in the other vector we put the in 
//degree of the target node.
int
Graph::getAssortativity()
{
	vector<double> srcDegree;
	vector<double> targetDegree;
	vector<double> rsrcDegree;
	vector<double> rtargetDegree;
	vector<double> oSrcDegree;
	vector<double> oTargetDegree;
	for(map<string,Vertex*>::iterator aIter=vertexList.begin();aIter!=vertexList.end();aIter++)
	{
		Vertex* sVertex=aIter->second;
		NINFO_MAP& outNeighbours=sVertex->getImmediateNeighbours();
		for(NINFO_MAP_ITER nmIter=outNeighbours.begin();nmIter!=outNeighbours.end();nmIter++)
		{
			Vertex* tVertex=nmIter->second;
			if(tVertex->getOutDegree()>0)
			{
				oSrcDegree.push_back(sVertex->getOutDegree());
				oTargetDegree.push_back(tVertex->getOutDegree());
			}
	//		else
	//		{
				srcDegree.push_back(sVertex->getOutDegree());
				targetDegree.push_back(tVertex->getInDegree());
				rsrcDegree.push_back(sVertex->getReachableOutDegree());
				rtargetDegree.push_back(tVertex->getReachableInDegree());
	//		}
		}
	}
	Distance d;
	outIncorrCoeff=d.computeCC(srcDegree,targetDegree);
	reachoutIncorrCoeff=d.computeCC(rsrcDegree,rtargetDegree);
	outOutcorrCoeff=d.computeCC(oSrcDegree,oTargetDegree);
	return 0;
}

int
Graph::showEdgeDegrees(const char* edgeFName)
{
	ofstream oFile(edgeFName);
	for(map<string,Vertex*>::iterator aIter=vertexList.begin();aIter!=vertexList.end();aIter++)
	{
		Vertex* sVertex=aIter->second;
		NINFO_MAP& outNeighbours=sVertex->getImmediateNeighbours();
		for(NINFO_MAP_ITER nmIter=outNeighbours.begin();nmIter!=outNeighbours.end();nmIter++)
		{
			Vertex* tVertex=nmIter->second;
			oFile << sVertex->getOutDegree() <<"\t" << tVertex->getInDegree() << endl;
		}
	}
	return 0;
}

//Here for each vertex we will compute the (a) closeness centrality, that is the
//extent to which a node is close to all other nodes, (b) betweenness centrality,
//that is how many shortest paths pass through a particular node.

int
Graph::computeDifferentNodeProperties()
{
	for(map<string,Vertex*>::iterator aIter=vertexList.begin();aIter!=vertexList.end();aIter++)
	{
		Vertex* sVertex=aIter->second;
		sVertex->computeCloseness(vertexList);
		sVertex->computeBetweenness(vertexList,totalPaths);
		sVertex->computeAvgPathLength();
		sVertex->computeSdPathLength();
		sVertex->computeAvgTargetIndegree();
		sVertex->computeAvgTFOutdegree();
		sVertex->computePureReachableNodes(vertexList);
	}
	computeTargetOverlap();
	computeAvgRegPathRatio();
	computeAverageBetweenness();
	computeAveragePureReachabilityDiff();
	return 0;
}

int
Graph::computeTargetOverlap()
{
	map<string,Vertex*> tfNodes;
	for(map<string,Vertex*>::iterator aIter=vertexList.begin();aIter!=vertexList.end();aIter++)
	{
		Vertex* v=aIter->second;
		if(v->getOutDegree()==0)
		{
			continue;
		}
		tfNodes[aIter->first]=v;
	}
	for(map<string,Vertex*>::iterator aIter=tfNodes.begin();aIter!=tfNodes.end();aIter++)
	{
		Vertex* tfVertex=aIter->second;
		tfVertex->computeOverlappingTargets(tfNodes);
	}
	return 0;
}

int
Graph::computeAvgRegPathRatio()
{
	avgRegPathRatio=0;
	sdRegPathRatio=0;
	avgRegulatibility=0;
	sdRegulatibility=0;
	int regulateeCnt=0;
	for(map<string,Vertex*>::iterator aIter=vertexList.begin();aIter!=vertexList.end();aIter++)
	{
		Vertex* aTarget=aIter->second;
		if(aTarget->getInDegree()==0)
		{
			continue;
		}
		aTarget->computeRegPathRatio();
		aTarget->computeRegulatibility();
		avgRegPathRatio=avgRegPathRatio+aTarget->getRegPathRatio();
		avgRegulatibility=avgRegulatibility+aTarget->getRegulatibility();
		regulateeCnt++;
	}
	avgRegPathRatio=avgRegPathRatio/(double) regulateeCnt;
	avgRegulatibility=avgRegulatibility/(double) regulateeCnt;
	
	for(map<string,Vertex*>::iterator aIter=vertexList.begin();aIter!=vertexList.end();aIter++)
	{
		Vertex* aTarget=aIter->second;
		if(aTarget->getInDegree()==0)
		{
			continue;
		}
		double diff=aTarget->getRegPathRatio()-avgRegPathRatio;
		sdRegPathRatio=sdRegPathRatio+(diff*diff);
		diff=aTarget->getRegulatibility()-avgRegulatibility;
		sdRegulatibility=sdRegulatibility+(diff*diff);
	}
	
	sdRegPathRatio=sqrt(sdRegPathRatio/((double)(regulateeCnt-1)));
	sdRegulatibility=sqrt(sdRegulatibility/((double)(regulateeCnt-1)));
	return 0;
}


int
Graph::getDegreeDist(bool degreeType,map<int,int>& degreeDist)
{
	for(map<string,Vertex*>::iterator aIter=vertexList.begin();aIter!=vertexList.end();aIter++)
	{
		Vertex* v=aIter->second;
		int d=0;
		if(degreeType)
		{
			d=v->getInDegree();
		}
		else
		{
			d=v->getOutDegree();
		}
		if(degreeDist.find(d)==degreeDist.end())
		{
			degreeDist[d]=1;
		}
		else
		{
			degreeDist[d]=degreeDist[d]+1;
		}
	}
	return 0;
}
