import unittest

from patronus.etc.schema import Document
from patronus.storing.module import SQLDocStore


class TestStoring(unittest.TestCase):
    def setUp(self):
        self.docs = [
            "We walk towards the Seam in silence. I don't like that Gale took a dig at Madge, but he's right, of course. The reaping system is unfair, with the poor getting the worst of it. You become eligible for the reaping the day you turn twelve. That year, your name is entered once. At thirteen, twice. And so on and so on until you reach the age of eighteen, the final year of eligibility, when your name goes into the pool seven times. That's true for every citizen in all twelve districts in the entire country of Panem.",
            "Just as the town clock strikes two, the mayor steps up to the podium and begins to read. It's the same story every year. He tells of the history of Panem, the country that rose up out of the ashes of a place that was once called North America. He lists the disasters, the droughts, the storms, the fires, the encroaching seas that swallowed up so much of the land, the brutal war for what little sustenance remained. The result was Panem, a shining Capitol ringed by thirteen districts, which brought peace and prosperity to its citizens. Then came the Dark Days, the uprising of the districts against the Capitol. Twelve were defeated, the thirteenth obliterated. The Treaty of Treason gave us the new laws to guarantee peace and, as our yearly reminder that the Dark Days must never be repeated, it gave us the Hunger Games.",
            "The rules of the Hunger Games are simple. In punishment for the uprising, each of the twelve districts must provide one girl and one boy, called tributes, to participate. The twenty-four tributes will be imprisoned in a vast outdoor arena that could hold anything from a burning desert to a frozen wasteland. Over a period of several weeks, the competitors must fight to the death. The last tribute standing wins.",
            "… That I’m ashamed I never tried to help her in the woods. That I let the Capitol kill the boy and mutilate her without lifting a finger.\nJust like I was watching the Games.\n\n I kick off my shoes and climb under the covers in my clothes. The shivering hasn’t stopped. Perhaps the girl doesn’t even remember me. But I know she does. You don’t forget the face of the person who was your last hope. I pull the covers up over my head as if this will protect me from the redheaded girl who can’t speak. But I can feel her eyes staring at me, piercing through walls and doors and bedding. I wonder if she’ll enjoy watching me die.",
            "I realize I do want to talk to someone about the girl. Someone who might be able to help me figure out her story. Gale would be my first choice, but it’s unlikely I’ll ever see Gale again. I try to think if telling Peeta could give him any possible advantage over me, but I don’t see how. Maybe sharing a confidence will actually make him believe I see him as a friend.\n Besides, the idea of the girl with her maimed tongue frightens me. She has reminded me why I’m here. Not to model flashy costumes and eat delicacies. But to die a bloody death while the crowds urge on my killer.",
            "\"I mean, one moment the sky was empty and the next it was there. It didn’t make a sound, but they saw it. A net dropped down on the girl and carried her up, fast, so fast like the elevator. They shot some sort of spear through the boy. It was attached to a cable and they hauled him up as well. But I’m certain he was dead. We heard the girl scream once. The boy’s name, I think. Then it was gone, the hovercraft. Vanished into thin air. And the birds began to sing again, as if nothing had happened.\"\n\n \"Did they see you?\" Peeta asked. \"I don’t know. We were under a shelf of rock,\" I reply. But I do know. There was a moment, after the birdcall, but before the hovercraft, where the girl had seen us. She’d locked eyes with me and called out for help. But neither Gale or I had responded.\n \"You’re shivering,\" says Peeta.",
            "They’re funny birds and something of a slap in the face to the Capitol. During the rebellion, the Capitol bred a series of genetically altered animals as weapons. The common term for them was muttations, or sometimes mutts for short. One was a special bird called a jabberjay that had the ability to memorize and repeat whole human conversations. They were homing birds, exclusively male, that were released into regions where the Capitol’s enemies were known to be hiding. After the birds gathered words, they’d fly back to centers to be recorded. It took people awhile to realize what was going on in the districts, how private conversations were being transmitted. Then, of course, the rebels fed the Capitol endless lies, and the joke was on it. So the centers were shut down and the birds were abandoned to die off in the wild.",
            "In school, they tell us the Capitol was built in a place once called the Rockies. District 12 was in a region known as Appalachia. Even hundreds of years ago, they mined coal here. Which is why our miners have to dig so deep.\n Somehow it all comes back to coal at school. Besides basic reading and math most of our instruction is coal-related. Except for the weekly lecture on the history of Panem. It’s mostly a lot of blather about what we owe the Capitol. I know there must be more than they’re telling us, an actual account of what happened during the rebellion. But I don’t spend much time thinking about it. Whatever the truth is, I don’t see how it will help me get food on the table.",
        ]

    def test_write(self):
        self.store = SQLDocStore()

        self.store.write([{"content": t} for t in self.docs])

        docs = self.store.get_all_documents()

        assert len(docs) == len(self.docs)

        assert self.store.get_document_count() == len(self.docs)


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
