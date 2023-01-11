/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
 #include <stdlib.h>
 #include <stdio.h>
 #include <string.h>
 #include <math.h>
 #include <cuda.h>
 
 #define MAX_THREADS_PER_BLOCK 512
 
 int no_of_nodes;
 int edge_list_size;
 FILE *fp;
 
 //Structure to hold a node information
 struct Node
 {
	 int starting;
	 int no_of_edges;
 };
 
 #include "kernel.cu"
 #include "kernel2.cu"
 
 void BFSGraph(int argc, char** argv);
 
 ////////////////////////////////////////////////////////////////////////////////
 // Main Program
 ////////////////////////////////////////////////////////////////////////////////
 int main( int argc, char** argv) 
 {
	 no_of_nodes=0;
	 edge_list_size=0;
	 BFSGraph( argc, argv);
 }
 
 void Usage(int argc, char**argv){
 
 fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);
 
 }
 ////////////////////////////////////////////////////////////////////////////////
 //Apply BFS on a Graph using CUDA
 ////////////////////////////////////////////////////////////////////////////////
 void BFSGraph( int argc, char** argv) 
 {
 
	 char *input_f;
	 if(argc!=2){
	 Usage(argc, argv);
	 exit(0);
	 }
	 
	 input_f = argv[1];
	 printf("Reading File\n");
	 //Read in Graph from a file
	 fp = fopen(input_f,"r");
	 if(!fp)
	 {
		 printf("Error Reading graph file\n");
		 return;
	 }
 
	 int source = 0;
 
	 fscanf(fp,"%d",&no_of_nodes);
 
	 int num_of_blocks = 1;
	 int num_of_threads_per_block = no_of_nodes;
 
	 //Make execution Parameters according to the number of nodes
	 //Distribute threads across multiple Blocks if necessary
	 if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	 {
		 num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		 num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	 }
 
	 // Allocate the Node list using unified memory 
	 Node* graph_nodes;
	 cudaMallocManaged( (void**) &graph_nodes, sizeof(Node)*no_of_nodes) ;
 
	 // Allocate the Mask using unified memory
	 bool* graph_mask;
	 cudaMallocManaged( (void**) &graph_mask, sizeof(bool)*no_of_nodes) ;
     
	 bool* updating_graph_mask;
	 cudaMallocManaged( (void**) &updating_graph_mask, sizeof(bool)*no_of_nodes) ;
 
	 // Allocate the Visited nodes array using unified memory
	 bool* graph_visited;
	 cudaMallocManaged( (void**) &graph_visited, sizeof(bool)*no_of_nodes) ;
 
	 // Allocate mem for the result in the managed memory
	 int* u_cost;
	 cudaMallocManaged( (void**) &u_cost, sizeof(int)*no_of_nodes);
	 for(int i=0;i<no_of_nodes;i++)
		 u_cost[i]=-1;
	 u_cost[source]=0;
	 
	 // Make a bool to check if the execution is over
	 bool *over;
	 cudaMallocManaged( (void**) &over, sizeof(bool));
 
	 printf("Allocate all unified memory\n");
 
	 int start, edgeno;   
	 // Initalize the unified memory
	 for( unsigned int i = 0; i < no_of_nodes; i++) 
	 {
		 fscanf(fp,"%d %d",&start,&edgeno);
		 graph_nodes[i].starting = start;
		 graph_nodes[i].no_of_edges = edgeno;
		 graph_mask[i]=false;
		 updating_graph_mask[i]=false;
		 graph_visited[i]=false;
	 }
    
	 //Read the source node from the file
	 fscanf(fp,"%d",&source);
	 source=0;
   
	 //Set the source node as true in the mask
	 graph_mask[source]=true;
	 graph_visited[source]=true;
 
	 fscanf(fp,"%d",&edge_list_size);
     //Allocate the Edge List using unified memory 
	 int* graph_edges;
	 cudaMallocManaged( (void**) &graph_edges, sizeof(int)*edge_list_size) ;
 
	 int id,cost;
	 for(int i=0; i < edge_list_size ; i++)
	 {
		 fscanf(fp,"%d",&id);
		 fscanf(fp,"%d",&cost);
		 graph_edges[i] = id;
	 }
 
	 if(fp)
		 fclose(fp);    
 
	 printf("Read File\n");
 
	 // setup execution parameters
	 dim3  grid( num_of_blocks, 1, 1);
	 dim3  threads( num_of_threads_per_block, 1, 1);
 
	 int k=0;
 
	 printf("Start traversing the tree\n");
	 bool stop;
	 // Call the Kernel untill all the elements of Frontier are not false
	 do
	 {
		 //if no thread changes this value then the loop stops
		 stop=false;
		 
         cudaMemcpy( over, &stop, sizeof(bool), cudaMemcpyHostToHost);   // Moving stop to over still using cudamemcpy but using HostToHost

		 Kernel<<< grid, threads, 0 >>>(graph_nodes, graph_edges, graph_mask, updating_graph_mask, graph_visited, u_cost, no_of_nodes);
		 // check if kernel execution generated and error

		 Kernel2<<< grid, threads, 0 >>>(graph_mask, updating_graph_mask, graph_visited, over, no_of_nodes);
		 // check if kernel execution generated and error
		 
		 cudaMemcpy( &stop, over, sizeof(bool), cudaMemcpyHostToHost);  // Moving over to stop still using cudamemcpy but using HostToHost

		 k++;
	 }
	 while(stop);
  
	 printf("Kernel Executed %d times\n",k);
 
	 //Store the result into a file
	 FILE *fpo = fopen("result.txt","w");
	 for(int i=0;i<no_of_nodes;i++)
		 fprintf(fpo,"%d) cost:%d\n",i,u_cost[i]);
	 fclose(fpo);
	 printf("Result stored in result.txt\n");
 
 
	 // cleanup memory
     cudaFree(over);
	 cudaFree(graph_nodes);
	 cudaFree(graph_edges); 
	 cudaFree(graph_mask);
	 cudaFree(updating_graph_mask);
	 cudaFree(graph_visited);
	 cudaFree(u_cost);
 }