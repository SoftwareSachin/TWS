import Image from "next/image";
import { Icon } from "@/components/chat-app/Icon";

export const VoiceButton = ({ isMicOn, setIsMicOn }) => {
  return (
    <div>
      <div className={"flex gap-2"}>
        {/*<Image alt={"mic"} width={100} height={100} src={"/images/mic_side_1.svg"}></Image1>*/}
        {/*<Image alt={"mic"} width={100} height={100} src={"/images/microphone.svg"}></Image>*/}
        {/*<Image alt={"mic"} width={100} height={100} src={"/images/mic_side_2.svg"}></Image>*/}
        <Icon iconName={"mic_side_1"}></Icon>
        <div onClick={setIsMicOn} className="mic cursor-pointer">
          <Image
            className={"mic-icon h-[60px] w-[60px]"}
            alt={"mic"}
            width={100}
            height={100}
            src={`/assets/chat/${isMicOn ? "mic_stop" : "microphone"}.svg`}
          ></Image>
          {isMicOn && <div className="mic-shadow"></div>}
        </div>
        <Icon iconName={"mic_side_2"}></Icon>
      </div>
      <div className={"text-[#767676]"}>Click and speak</div>
    </div>
  );
};
