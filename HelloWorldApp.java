
package helloworldapp;

import org.json.*;
import java.net.*;
public class HelloWorldApp {

    public static void main(String[] args) {
        
try {
URI uri = new URI("file:///C:/Users/Josh/Desktop/UROPScheme/set1.txt");
JSONTokener tokener = new JSONTokener(uri.toURL().openStream());

//10,000 tweets per file
//retweetee's followers
int[] RTFollowersArr;
RTFollowersArr = new int[10000];
//tweeter's followers
int[] TFollowersArr;
TFollowersArr = new int[10000];

int currentRTUserCount = 0;
int currentTUserCount = 0;

////////////////////////////////////////////////
//loop all objects (10,000 lines)
for(int i = 0; i < 10000; i++) {
    
 //object
JSONObject base = new JSONObject(tokener);

//if retweet
if (base.has("retweeted_status")) {
    JSONObject root = base.getJSONObject("retweeted_status");
    JSONObject rootUser = root.getJSONObject("user");
    
    int rootFollowers = rootUser.getInt("followers_count");
    int rootFriends = rootUser.getInt("friends_count");
    int rootTweets = rootUser.getInt("statuses_count");
    RTFollowersArr[currentRTUserCount] = rootFollowers;
    
    System.out.println("Retweetee's followers: " + rootFollowers);
    System.out.println("Retweetee's friends: " + rootFriends);
    System.out.println("Retweetee's tweets: " + rootTweets);
    currentRTUserCount++;
}
//if original tweet
else {
    JSONObject rootUser = base.getJSONObject("user");
    
    int rootFollowers = rootUser.getInt("followers_count");
    int rootFriends = rootUser.getInt("friends_count");
    int rootTweets = rootUser.getInt("statuses_count");
    TFollowersArr[currentTUserCount] = rootFollowers;
    
    System.out.println("Tweeter's followers: " + rootFollowers);
    System.out.println("Tweeter's friends: " + rootFriends);
    System.out.println("Tweeter's tweets: " + rootTweets);
    currentTUserCount++;
}

}
///////////////////////////////////////////////

int sumOfRTFollowers, sumOfTFollowers;
sumOfRTFollowers = 0;
sumOfTFollowers = 0;

for(int j=0; j < RTFollowersArr.length; j++){
    sumOfRTFollowers += RTFollowersArr[j];
}
for(int k=0; k < TFollowersArr.length; k++){
    sumOfTFollowers += TFollowersArr[k];
}

double avgRTFollowers, avgTFollowers;
avgRTFollowers = 1.0d * sumOfRTFollowers / RTFollowersArr.length;
avgTFollowers = 1.0d * sumOfTFollowers / TFollowersArr.length;

System.out.println("Average retweetee's followers: " + avgRTFollowers);
System.out.println("Average original tweeter's followers: " + avgTFollowers);

}
catch(java.net.URISyntaxException|java.io.IOException|org.json.JSONException e){
    System.out.println(e);
}

    }
}
