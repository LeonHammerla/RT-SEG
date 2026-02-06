import time

from rt_segmentation import (RTLLMOffsetBased,
                             RTLLMForcedDecoderBased,
                             RTLLMSegUnitBased,
                             RTRuleRegex,
                             RTNewLine,
                             RTPRMBase,
                             bp,
                             sdb_login,
                             load_prompt,
                             load_example_trace, RTLLMSurprisal, RTLLMEntropy, RTLLMTopKShift, RTLLMFlatnessBreak,
                             export_gold_set, export_rf_data_gold_set, upload_rf_data)




if __name__ == "__main__":
    # export_rf_data_gold_set()
    upload_rf_data()
    """trace = 'We have a relativistic scenario: In ejecta\'s rest frame, the Galaxy center moves towards it at speed 60,000 km/s. That speed is 60,000 km/s = 60,000/299,792 ≈ 0.2003 c. So gamma ~ 1/sqrt(1-β^2). Let\'s compute.\n\nβ = 60000 km/s = 60000/299792.458 km/s. 299792.458 km/s is speed of light c. Let\'s calculate:\n\n60000/299792.458 = approx 0.2002.\n\nCompute more precisely: 60000/299792.458 = 0.200000? Wait: 300,000 km/s approximate c. But let\'s compute using 299,792.458 km/s.\n\n60000 / 299792.458 = 0.200. Let\'s approximate.\n\n299,792.458 * 0.2 = 59,958.4916. So difference 41.5084. So ratio 0.200138? Let\'s compute more precisely:\n\n0.200138 = 0.200138 * 299,792.458 = 59,957? Let\'s compute: 0.200138 * 299,792.458 = 59,957.9? Wait I\'m off.\n\nLet\'s do high precision: β = 60000 / 299792.458 = ?\n\nWe can compute: 299,792.458 * 0.200 = 59,958.4916. So 60000 - 59,958.4916 = 41.5084. So ratio difference 41.5084 / 299,792.458 = 0.0001384. So β = 0.2001384. Good.\n\nThus β^2 = 0.040055. (0.2001384^2). Let\'s compute: 0.2001384^2: 0.2001384*0.2001384.\n\n0.2001384 * 0.2001384 = (0.2 + 0.0001384)*(0.2+0.0001384) = 0.04 + 2*0.2*0.0001384 + (0.0001384)^2 = 0.04 + 0.00005536 + 1.917...e-8 ≈ 0.040055378. Good.\n\nThus gamma = 1/sqrt(1 - β^2) = 1/sqrt(1 - 0.040055378) = 1/sqrt(0.959944622) ≈ 1/0.97980 = 1.0207? Wait sqrt(0.95994) ~ 0.9798. Inverse ~1.0207. Let\'s compute precisely:\n\nsqrt(0.959944622) =? We can approximate: sqrt(0.96) = 0.979795. Slightly less. So gamma ≈ 1.0206.\n\nThus the ejecta\'s proper time τ = 50 s. In galaxy rest frame, time dilation: t = gamma * τ ≈ 1.0206 * 50 s = 51.03 s. So about 51 s.\n\nDistance traveled by ejecta in galaxy rest frame: v = 60000 km/s (as measured in galaxy rest). But from ejecta perspective, galaxy moving at that speed. So the relative velocity is same. The proper distance? In galaxy rest frame, ejecta moves at 60,000 km/s. So distance = v * t = 60000 km/s * 51.03 s ≈ 3,061,800 km.\n\nLet\'s compute precisely: 60000 km/s * 51.03 s = 3,061,800 km. (Actually 60000*51 = 3,060,000. plus 60000*0.03=1800. So 3,061,800). Options: 3,060,000 km; 2,940,000 km; 2,880,000 km; 3,000,000 km. 3,061,800 km not in list. 3,060,000 km is very close. So answer likely option 0: 3,060,000 km.\n\nBut we should check if we need to account for relativistic addition of distances? The problem: "From the reference frame of the ejecta, the center of the Galaxy is moving at 60 000 km/s towards it. What distance the ejecta travels in the Galaxy reference frame when 50 seconds pass in the ejecta reference frame?" This is a simple Lorentz transformation: In ejecta rest frame, the galaxy moves toward it at 60,000 km/s. We want the distance traveled by the ejecta relative to galaxy when 50s of proper time passes in ejecta frame. Since velocities are same magnitude. But due to time dilation, the coordinate time is t = γ τ. So the distance traveled in galaxy rest = v t = v γ τ.\n\nThus the answer is 3,061,800 km ≈ 3,060,000 km. Option 0. We\'ll choose 0.\n\nBut check that we might need to use relativistic addition? No, because we only transform coordinates between two frames: one moving relative to the other at 0.200 c. In ejecta frame, the galaxy moves at 0.200 c. In galaxy frame, the ejecta moves at 0.200 c. So velocities same magnitude. So distance is just velocity times coordinate time.\n\nThus answer 0.\n\nDouble-check options: 3 060 000 km. Yes.\n\nTherefore explanation: compute gamma, find coordinate time, multiply by velocity. Provide explanation.\n\nThus final JSON: {"answer": 0}.'
    res = RTPRMBase._segment(
        trace=trace,
        seg_base_unit="clause",
        problem='A supernova occurred in our Galaxy. Part of the supernova ejecta is moving directly towards the center of the Galaxy. Both the ejecta part and the Galaxy are inertial frames of reference. From the reference frame of the ejecta, the center of the Galaxy is moving at 60 000 km/s (sixty thousand kilometers per second) towards it.\nWhat distance the ejecta travels in the Galaxy reference frame when 50 seconds pass in the ejecta reference frame?\n\n(Note: Gravity can be ignored, and rotation and revolution of the Galaxy can be ignored.)'
        )

    for r in res[0]:
        print(10*"=")
        print(trace[r[0]:r[1]])

    print("res: ", res[0])"""