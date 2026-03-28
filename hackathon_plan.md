# Objective

Make SCP-173 from SCP Containment Breach

i.e. if it sees a person that has their eyes closed, it makes a beeline towards that person

unanswered questions:
- what if two people have their eyes closed?
- how does it navigate?
- how do we send get it to move?

tentative plan:

1. figure out how camera data is sent over to us from the computer.
   if we manage to get the raw camera stream, we can run a model offline that allows us to
   a. do monocular depth estimation
   b. figure out whose eyes are closed

2. once that's done, we want to set a path towards that person. this means streaming motion commands
   to the model. so figure out how to do that

3. then go from deploying it on a PC to deploying it on the comma v4 (stretch)


# FAQ

1. How to SSH into the thing
a. only one person can SSH in at once time. Be on `unifi` WIFI and ssh comma@192.168.63.120. 
   you should have a github account that you logged into `connect.comma.ai` with
   and also set the ssh username on the comma v4 to your github username
