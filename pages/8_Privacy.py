import streamlit as st 

st.set_page_config(page_title="Picanalitika | Privacy Policy", layout="wide")

hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div.vis-configuration-wrapper{display: block; width: 400px;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


st.title('Privacy Policy')

st.markdown("At Picanalitika, we know that you care about how your personal information is used and shared, and that is why we are committed to respecting your privacy and to helping protect the security of your information. In this Privacy Policy (“Policy”), we describe the information that we collect about you for and/or through the use of our services (our “Services”), and how we use and disclose that information."

"By using any of our Services, you agree that your information will be handled as described in this Policy. Your use of our Services, and any dispute over privacy, is subject to this Policy and the agreement signed between us (“the Agreement”), including its applicable limitations on damages and the resolution of disputes.")

st.header('What information do we collect about you and why?')
st.markdown("We process your personal information to fulfill our contractual obligations to provide you with our Services. We need your data to create and maintain your account to our services, to create logins for you and to contact you in case you need support. We collect this data directly from you when you sign up as our Customer."

"We collect the following categories of your personal data:"

"Name, email address, phone number, your username to our Services."

"Should you wish to connect your social accounts to our Services, we also require your username and tokens for your social media profiles."

"We also collect data about your use of our Services automatically according to our Cookie Statement."

"Picanalitika follows all Terms of Service for social media sites ingested into our platform.")

st.header('Social Network Privacy Policy Links')
st.markdown("Twitter- https://twitter.com/en/privacy")
st.markdown("YouTube- https://policies.google.com/privacy")

st.markdown("https://developers.google.com/youtube/terms/developer-policies#a.-api-client-terms-of-use-and-privacy-policies")
st.markdown("Facebook and Instagram-https://www.facebook.com/policy.php")
st.markdown("Tiktok: https://www.tiktok.com/legal/page/row/privacy-policy/en")

st.header('How we use your information')
st.markdown("Information about our customers is an integral part of our business. We use your information, including your personal information, for the following purposes:"

"To provide our Services to you, to communicate with you about your use of our Services, to respond to your inquiries, to manage your account, to send you information that you have requested, and for other customer service purposes."
"For marketing and advertising purposes. For example, as permitted by applicable law, we may occasionally contact you by e-mail, postal mail, or telephone to provide you with information about other Picanalitika services that might be of interest to you."
"To tailor the content and information that we may send or display to you, to offer location customization, and personalized help and instructions, and to otherwise personalize your experiences while using the Site or our Services."
"To better understand how users access and use our Services, both on an aggregated and individualized basis, in order to improve our Services and respond to user desires and preferences, and for other Picanalitika service research and analytical purposes.")
st.header('When We Disclose Your Information')
st. markdown("We do not share your information with non-affiliated third parties for their own marketing purposes. We do share your information as follows:"

"To Our Affiliates. As permitted by applicable law, we may share your personal information with our affiliated entities. Those entities may market their products and services to you, but their use of your personal information will be governed by this privacy policy."
"Sales Channel Partners and Distributors: We may provide information about you to carefully screened entities that work on our behalf, such as our Marketing services suppliers, newsletter distributors or other service partners, to market Picanalitika products and services. Only trusted companies who require the use of said information for business purposes are given access to it."
"Service Providers. We may share your information with other third party service providers to assist us in providing our Services to you, responding to your requests, and/or to provide you with the partner services you have purchased."
"We also may share your information in the following circumstances:"

"Business Transfers. If we are acquired by or merged with another company, if substantially all of our assets are transferred to another company, or as part of a bankruptcy proceeding, we may transfer the information we have collected from you to the other company. We also may transfer our rights under any customer agreement we have with you."
"In Response to Legal Process. We also may disclose the information we collect from you in order to comply with the law, a judicial proceeding, court order, or other legal process, such as in response to a court order or a subpoena."
"To Protect Us and Others. We also may disclose the information we collect from you where we believe it is necessary to investigate, prevent, or take action regarding illegal activities, suspected fraud, situations involving potential threats to the safety of any person, violations of the Agreement or this Policy, or as evidence in litigation in which we are involved."
"Aggregate and Anonymous Information. We may share aggregate or anonymous information about users with third parties for research or similar purposes."
"Legal Purposes. To enforce our Terms of Use; for example, to protect the security of our Services, as required by law or when we believe that disclosure is necessary to protect our rights.")

st.header('Legal Basis for Processing')
st.markdown("We have legal basis to collect, use and share your information. You also have choices about our use of your information as explained further below."

"We will only collect and process your information when we have legal basis to do so. Legal basis includes consent (where you have given consent), contract (where processing is necessary to fulfil contractual obligations (e.g. to provide you with the Services you have purchased)) and legitimate interest."

"If you have any questions about the legal basis we use to process your information, please contact us at privacy@Picanalitika.com.")

st.header('Data retention')
st.markdown("We retain your information while you use our Services or as needed to provide you the Services. This includes data you or others provided to us and data generated or inferred from your use of our Services. Your account and data related to your use of the Services, excluding anonymized data, will be deleted 6 months after the expiry of the Service agreement between us.")

st.header('Security')
st.markdown("Picanalitika takes security of all data, especially your personal information, seriously. We have done our best to protect your information from any unauthorized access or loss with implemented security features and procedures. You should however be aware that the transmission of information via the Internet is never completely secure. You should also take necessary steps to protect against unauthorized access to your password, phone, and computer by, among other things, signing off after using a shared computer, choosing a robust password that nobody else knows or can easily guess, and keeping your log-in and password private. We are not responsible for any lost, stolen, or compromised passwords or for any activity on your account via unauthorized password activity.")

st.header('Transfers to countries that do not provide adequate level of protection')
st.markdown("Picanalitika is a global company and information that we collect from or about you within the European Economic Area (“EEA”) may be transferred across international borders and to countries outside of the EEA that do not have laws providing an adequate level of data protection. We as the Data Controller (or Data Exporter) will do our best to ensure that your information is protected no matter where it is transferred to. To ensure adequate protection we enter into Data Processing Agreements with our suppliers processing your information outside of the EEA. When necessary we utilize standard contract clauses approved by the European Commission or rely on other legally-provided mechanisms (such as the EU-US or Swiss-US Privacy Shield) to ensure the adequate protection of your information.")

st.header('Opting out of outreach from Picanalitika')
st.markdown("We may send periodic promotional or informational emails to you. You may opt-out of such communications by following the opt-out instructions contained in the e-mail or by contacting us at privacy@Picanalitika.com. If you opt-out of receiving emails about recommendations or other information we think may interest you, we may still send you e-mails about your account or any services you have requested or received from us.")

st.header('Your rights as a data subject')
st.markdown("You as a data subject have many possibilities to impact how your personal information is being processed.  You can:"

"Request access to your information we process."
"Request for your information to be deleted or for you to be forgotten (right to erasure)"
"Request us to correct or change your information. As a customer you may also do this by logging in to your account."
"Limit, restrict or object to our use of your information"
"Access your information and/or receive it in a machine readable form"
"Subject to applicable law lodge a complaint with your local data protection authority or the Berlin Data Protection Authority, which is Picanalitika’s lead supervisory authority in the European Union."
"Please note that if you request us to remove your information, we may retain some of the information for specific reasons, such as to resolve disputes, troubleshoot problems, and as required by law. Furthermore, some information is never completely removed from our databases because of technical constraints and the fact that we regularly back up our systems. Therefore, not all of your personal information will ever be completely removed from our databases.")

st.header('Identity and contact details of the Data Controller')
st.markdown("The service contract signed between us and you includes the details of which Picanalitika entity is the Data Controller of your information. You will find further contact details in your contract or here."

"If you have any questions about this policy or the use of your personal information, you are always welcome to contact us at privacy@Picanalitika.com. You will also reach our Data Protection Officer at the same address.")

st.header('Changes to the Privacy Policy')
st.markdown("We may change this Privacy Policy from time to time due to changed or updated legislation and/or business standards. All changes to this Privacy Policy are posted on this page and we  encourage you to review our Privacy Policy regularly to stay informed.")