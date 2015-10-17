/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nowcoder.course;

import java.io.IOException;
import java.util.StringTokenizer;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class Join {

  public static String CTRL_A = "\u0001";

  public static class MapperOne 
       extends Mapper<Object, Text, IntWritable, Text>{
    
    private String tag = "0"; 
    private IntWritable k = new IntWritable();
    private Text v = new Text();
     
    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
      String[] vals = value.toString().split("\t", -1);
      if(2 != vals.length) {
        context.getCounter("Error", "Input Data Error").increment(1);
        return;
      }
      k.set(Integer.valueOf(vals[0]));
      v.set(tag + CTRL_A + vals[1]);
      context.write(k, v);
    }
  }
  
  public static class MapperTwo
       extends Mapper<Object, Text, IntWritable, Text>{
    
    private String tag = "1"; 
    private IntWritable k = new IntWritable();
    private Text v = new Text();
     
    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
      String[] vals = value.toString().split("\t", -1);
      if(2 != vals.length) {
        context.getCounter("Error", "Input data error").increment(1);
        return;
      }
      k.set(Integer.valueOf(vals[0]));
      v.set(tag + CTRL_A + vals[1]);
      context.write(k, v);
    }
  }

  public static class Reduce 
       extends Reducer<IntWritable,Text,IntWritable,Text> {
    
    private ArrayList<String> dataFromOne = new ArrayList();
    private ArrayList<String> dataFromTwo = new ArrayList();
    private Text v = new Text();

    public void reduce(IntWritable key, Iterable<Text> values, 
                       Context context
                       ) throws IOException, InterruptedException {
      dataFromOne.clear();
      dataFromTwo.clear();
      for (Text val : values) {
        String[] vals = val.toString().split(CTRL_A, -1);
        if(2 != vals.length) {
          context.getCounter("Error", "Reduce input error").increment(1);
          return;
        }
        if(vals[0].equals("0")) {
          dataFromOne.add(vals[1]);
        } else if(vals[0].equals("1")) {
          dataFromTwo.add(vals[1]);
        }
      }
      // output
      for(String v1 : dataFromOne) {
        for(String v2 : dataFromTwo) {
          v.set(v1 + "\t" + v2);
          context.write(key, v);
        }
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
    if (otherArgs.length < 3) {
      System.err.println("Usage: wordcount <in> [<in>...] <out>");
      System.exit(2);
    }
    Job job = Job.getInstance(conf, "join");
    job.setJarByClass(Join.class);
    //job.setMapperClass(TokenizerMapper.class);
    job.setReducerClass(Reduce.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(Text.class);
    MultipleInputs.addInputPath(job, new Path(otherArgs[0]), TextInputFormat.class, MapperOne.class);
    MultipleInputs.addInputPath(job, new Path(otherArgs[1]), TextInputFormat.class, MapperTwo.class);
    FileOutputFormat.setOutputPath(job, new Path(otherArgs[otherArgs.length - 1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
