import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from packaging.version import Version
from typing import Callable
import statistics
import matplotlib.pyplot as plt
import numpy as np
import random
import logging

logging.basicConfig(filename='simtim.log',filemode='w',level=logging.DEBUG,format='%(levelname)s:%(message)s')
#logging.getLogger().addHandler(logging.StreamHandler())


#--------------------
# basic classes
#--------------------


class Node:
    def __init__(self, id: str) -> None:
        self.id=id
        self.propertyValues={}
        self.outgoingLinks=[]
    def __str__(self) -> str:
        return "Node[id:"+self.id+",properties:"+str(self.propertyValues)+",outLinks:["+','.join(map(str, self.outgoingLinks))+"]]"

class Link:
    def __init__(self, startNode: Node, endNode: Node) -> None:
        self.startNode=startNode
        self.endNode=endNode
    def __str__(self) -> str:
        return "Link["+self.startNode.id+"->"+self.endNode.id+"]"

class AccessNode(Enum):
    NONE=auto()
    VISIBLE=auto()
    USER=auto()
    ADMIN=auto()

class AccessLink(Enum):
    NONE=auto()
    VISIBLE=auto()

class Actor:
    def __init__(self, name: str, doStep: Callable[[],None], capacity: int = None) -> None:
        self.name=name
        self.doStep=doStep
        self.capacity=capacity
        self.incurredCost=0
        self.nodeAccess={}
        self.linkAccess={}
    def __str__(self) -> str:
        nodeAccessStr=",".join([node.id+":"+access.name for node,access in self.nodeAccess.items()])
        linkAccessStr=",".join([str(link)+":"+access.name for link,access in self.linkAccess.items()])
        return "Actor[name:"+self.name+",capacity:"+str(self.capacity)+",cost:"+str(self.incurredCost)+",nodeAccess:["+nodeAccessStr+"],linkAccess:["+linkAccessStr+"]]"

class Action(ABC):
    def __init__(self, name: str, cost: float, duration: float, probability: float, actor: Actor) -> None:
        self.name=name
        self.cost=cost
        self.duration=duration
        self.probability=probability
        self.actor=actor

    @abstractmethod
    def precondition(self) -> bool:
        pass

    @abstractmethod
    def apply(self) -> None:
        pass

class NodeAction(Action):
    def __init__(self, name: str, cost: float, duration: float, probability: float, actor: Actor, node: Node) -> None:
        super().__init__(name,cost,duration,probability,actor)
        self.node=node

class LinkAction(Action):
    def __init__(self, name: str, cost: float, duration: float, probability: float, actor: Actor, link: Link) -> None:
        super().__init__(name,cost,duration,probability,actor)
        self.link=link

class WaitAction(Action):
    def __init__(self, duration: float, actor: Actor) -> None:
        super().__init__("Wait",0,duration,1,actor)

    def precondition(self) -> bool:
        return True

    def apply(self) -> None:
        pass

class DetectAction(Action):
    def __init__(self, duration: float, actor: Actor, action : Action, infraElement : object) -> None:
        super().__init__("Detection",0,duration,1,actor)
        self.action=action
        self.infraElement=infraElement

    def precondition(self) -> bool:
        return True

    def apply(self) -> None:
        pass


#--------------------
# action handling
#--------------------


class ActionItem:
    def __init__(self, actor: Actor, action: Action, infraElement: object, startTime: float, endTime: float) -> None:
        self.actor=actor
        self.action=action
        self.infraElement=infraElement #node or link
        self.startTime=startTime
        self.endTime=endTime

    def isSameAttack(self, actor: Actor, action: Action, infraElement: object) -> bool:
        return self.actor==actor and type(self.action)==type(action) and self.infraElement==infraElement

class ActionHandler:
    def __init__(self, maxTime: float) -> None:
        self.maxTime=maxTime
        self.actionQueue=[]
        self.now=0

    def nrActionsOfActor(self, actor: Actor) -> int:
        return sum([(actionItem.actor==actor) for actionItem in self.actionQueue])

    def addActionItem(self, actor: Actor, action: Action, infraElement: object, startTime: int = None) -> None:
        if(startTime==None):
            startTime=self.now
        #logging.info('Adding new action '+action.name+' from '+actor.name+' at '+str(self.now))
        if(actor.capacity is not None):
            if(self.nrActionsOfActor(actor)>=actor.capacity):
                #logging.info('Capacity of '+actor.name+' does not allow starting new action')
                return
        preconditionFulfilled: bool
        try:
            preconditionFulfilled=action.precondition()
        except:
            preconditionFulfilled=False
        if(preconditionFulfilled==False):
            #logging.info('Precondition not satisfied, action not added')
            return
        assert(startTime>=self.now)
        endTime=startTime+action.duration
        actor.incurredCost+=action.cost
        newActionItem=ActionItem(actor,action,infraElement,startTime,endTime)
        inserted=False
        for i in range(len(self.actionQueue)):
            if(self.actionQueue[i].endTime>endTime):
                self.actionQueue.insert(i,newActionItem)
                inserted=True
                break
        if(not inserted):
            self.actionQueue.append(newActionItem)
        detectProb=getDetectionProbability(action,infraElement)
        if(random.random()<detectProb):
            detectionTime=random.random()*action.duration
            detectionEvent=DetectAction(detectionTime,defender,action,infraElement)
            self.addActionItem(defender,detectionEvent,infraElement)

    def doNextActionItem(self) -> None:
        if(not self.actionQueue):
            logging.info('Action queue empty')
            return
        actionItem=self.actionQueue.pop(0)
        newNow=actionItem.endTime
        #logging.info('Popping action '+actionItem.action.name+' from '+actionItem.actor.name+' at '+str(newNow))
        self.now=newNow
        if(random.random()<actionItem.action.probability):
            logging.info('Successful action '+actionItem.action.name+' from '+actionItem.actor.name+' at '+str(newNow))
            actionItem.action.apply()
            if(actionItem.actor!=defender):
                if(type(actionItem.infraElement)==Node):
                    targetNode=actionItem.infraElement
                else:
                    targetNode=actionItem.infraElement.endNode
                actionItem.actor.incurredCost-=getOneOffGain(actionItem.action,targetNode)
                defender.incurredCost+=getOneOffDamage(actionItem.action,targetNode)
            self.enforcePreconditions()
            actionItem.actor.doStep(actionItem,True)
        else:
            #logging.info('Action not performed (reason: randomness)')
            actionItem.actor.doStep(actionItem, False)

    def enforcePreconditions(self) -> None:
        for actionItem in self.actionQueue:
            preconditionFulfilled: bool
            try:
                preconditionFulfilled=actionItem.action.precondition()
            except:
                preconditionFulfilled=False
            if(preconditionFulfilled==False):
#            if(not (actionItem.action.precondition())):
                self.actionQueue.remove(actionItem)
                #logging.info('Action removed because precondition not met anymore: '+str(actionItem))

    def containsAttack(self, actor: Actor, action: Action, infraElement: object) -> bool:
        result=False
        for actionItem in self.actionQueue:
            if(actionItem.isSameAttack(actor,action,infraElement)):
                result=True
                break
        return result

    def doLoop(self) -> None:
        while(self.actionQueue and self.now<self.maxTime):
            self.doNextActionItem()

def getDetectionProbability(action: Action, infraElement: object) -> float:
    result=0.0
    if(type(action)==CompromiseTapestry):
        result=0.8
    return result


#--------------------
# specific actions
#--------------------


class CompromiseTapestry(NodeAction):
    def __init__(self, actor: Actor, node: Node) -> None:
        super().__init__("Tapestry attack", 300, tapestryDuration, 0.8, actor, node)

    def precondition(self) -> bool:
        return self.node.propertyValues["WebApp framework name"]=="Apache Tapestry" and self.node.propertyValues["WebApp framework version"] in ("5.4.5","5.5.0","5.6.2","5.7.0") and self.actor.nodeAccess[self.node]!=AccessNode.NONE

    def apply(self) -> None:
        self.actor.nodeAccess[self.node]=AccessNode.ADMIN

class PortScan(NodeAction):
    def __init__(self, actor: Actor, node: Node) -> None:
        super().__init__("Port scan", 0, 2, 0.9, actor, node)

    def precondition(self) -> bool:
        return self.actor.nodeAccess[self.node]==AccessNode.ADMIN

    def apply(self) -> None:
        for link in self.node.outgoingLinks:
            self.actor.linkAccess[link]=AccessLink.VISIBLE

class CompromiseRemoteMySQL(LinkAction):
    def __init__(self, actor: Actor, link: Link) -> None:
        super().__init__("Remote attack on MySQL", 800, remoteCompromiseDuration, 0.8, actor, link)

    def precondition(self) -> bool:
        node=self.link.endNode
        return node.propertyValues["DBMS name"]=="MySQL" and Version(node.propertyValues["DBMS version"])>=Version("8.0.0") and Version(node.propertyValues["DBMS version"])<=Version("8.0.28") and self.actor.linkAccess[self.link]==AccessLink.VISIBLE

    def apply(self) -> None:
        self.actor.nodeAccess[self.link.endNode]=AccessNode.ADMIN

class UpgradeMySQL(NodeAction):
    def __init__(self, actor: Actor, node: Node) -> None:
        super().__init__("Upgrading MySQL to new version", 500, defenseDuration, 0.9, actor, node)

    def precondition(self) -> bool:
        return self.node.propertyValues["DBMS name"]=="MySQL" and Version(self.node.propertyValues["DBMS version"])<Version("9.0.1") and self.actor.nodeAccess[self.node]==AccessNode.ADMIN

    def apply(self) -> None:
        self.node.propertyValues["DBMS version"]="9.0.1"


#--------------------
# damage and gain
#--------------------


def getOneOffDamage(action : Action, node : Node) -> float:
    result=0.0
    if(type(action)==CompromiseRemoteMySQL):
        result=150000.0
    return result

def getOneOffGain(action : Action, node : Node) -> float:
    result=0.0
    if(type(action)==CompromiseRemoteMySQL):
        result=100000.0
    return result

def getMarginalDamage(actor : Actor, node : Node) -> float:
    result=0.0
    return result

def getMarginalGain(actor : Actor, node : Node) -> float:
    result=0.0
    return result


#--------------------
# simple strategies
#--------------------

attackPerformedSuccessfully=set()

def doStepAttacker(prevActionItem : ActionItem, prevSuccess : bool) -> None:
    # print(attacker)
    if(prevSuccess):
        attackPerformedSuccessfully.add((prevActionItem.infraElement,type(prevActionItem.action)))
    for node in nodes:
        if(attacker.nodeAccess[node]!=AccessNode.NONE):
            for attackNodeAction in attackNodeActions:
                if (not ((node,attackNodeAction) in attackPerformedSuccessfully)):
                    attack=attackNodeAction(attacker,node)
                    if(aHandler.containsAttack(attacker, attack, node)==False):
                        aHandler.addActionItem(attacker, attack, node)
    for link in links:
        if(attacker.linkAccess[link]==AccessLink.VISIBLE):
            for attackLinkAction in attackLinkActions:
                if not ((link,attackLinkAction) in attackPerformedSuccessfully):
                    attack=attackLinkAction(attacker, link)
                    if(aHandler.containsAttack(attacker, attack, link)==False):
                        aHandler.addActionItem(attacker, attack, link)

def doStepDefender(prevActionItem : ActionItem, prevSuccess : bool) -> None:
    if(prevActionItem and type(prevActionItem.action)==DetectAction):
        for node in nodes:
            for defenseNodeAction in defenseNodeActions:
                defense=defenseNodeAction(defender,node)
                aHandler.addActionItem(defender, defense, node)
        for link in links:
            for defenseLinkAction in defenseLinkActions:
                defense=defenseLinkAction(defender, link)
                aHandler.addActionItem(defender, defense, link)


#--------------------
# test scenario
#--------------------


tapestryDuration=40
remoteCompromiseDuration=40
defenseDuration=80
#experiment="Tapestry attack"
#experiment="remote MySQL attack"
experiment="upgrading MySQL"
durations=(10,40,70,100,130)
allResults=[]
means=[]
stds=[]
for duration in durations:
    if(experiment=="Tapestry attack"):
        tapestryDuration=duration
    if(experiment=="remote MySQL attack"):
        remoteCompromiseDuration=duration
    if(experiment=="upgrading MySQL"):
        defenseDuration=duration
    results=[]
    for i in range(1000):
        n1=Node("n1")
        n1.propertyValues={"Name":"Web server", "WebApp framework name":"Apache Tapestry", "WebApp framework version":"5.7.0", "Endpoint protection":"Sophos"}
        n2=Node("n2")
        n2.propertyValues={"Name":"Database", "DBMS name":"MySQL", "DBMS version":"8.0.28", "Data sensitivity":"High", "Data amount":"10 million"}
        l=Link(n1,n2)
        n1.outgoingLinks=[l]
        nodes=[n1,n2]
        links=[l]

        defender=Actor("defender",doStepDefender,1)
        defender.nodeAccess={n1:AccessNode.ADMIN, n2:AccessNode.ADMIN}
        defender.linkAccess={l:AccessLink.VISIBLE}
        attacker=Actor("attacker",doStepAttacker)
        attacker.nodeAccess={n1:AccessNode.VISIBLE, n2:AccessNode.NONE}
        attacker.linkAccess={l:AccessLink.NONE}
        attackNodeActions=[CompromiseTapestry,PortScan]
        attackLinkActions=[CompromiseRemoteMySQL]
        defenseNodeActions=[UpgradeMySQL]
        defenseLinkActions=[]

        aHandler = ActionHandler(168)
        doStepAttacker(None,False)
        doStepDefender(None,False)
        aHandler.doLoop()
        results.append(defender.incurredCost)
    allResults.append(results)
    means.append(statistics.mean(results))
    stds.append(statistics.stdev(results))
    print(defenseDuration,sum(results)/len(results))

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots()
bp = ax.violinplot(allResults,durations,points=100,widths=20,showmeans=True)
ax.set_xlabel("duration of "+experiment)
if(experiment=="upgrading MySQL"):
    ax.set_ylabel("damage [USD]")
else:
    ax.yaxis.set_ticklabels([])
plt.savefig(experiment.replace(' ','_')+'.pdf',bbox_inches='tight')
plt.show(block=True)
